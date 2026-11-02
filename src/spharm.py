import numpy as np
import igl
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import pyshtools as pysh
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import src.utils as utils

jax.config.update("jax_enable_x64", True)     # L-BFGS line searches need double precision


# Spherical-map setting for the shape pipeline (compute_master_npz): long input edges split, then
# 5000 faces, Delaunay edge flips and tangential smoothing; then radial start when it has no folds,
# cotangent edges, signed areas per vertex, L-BFGS on up to 10 outer iterations, mu capped at 1e3.
# Decimation leaves needle triangles and folded slivers, and these give some near-round organoids
# a local map defect that shows up as a high complexity score.
N_FACES      = 5000
PRIMAL_STEPS = 500       # L-BFGS iterations per outer iteration, at most
OUTER_ROUNDS = 10        # outer iterations, at most
ENERGY_TOL   = 1e-3      # stop once the energy changes by less than this between outer iterations
AREA_TOL     = 1e-3      # and the worst relative area error around a vertex is below this
SPLIT_FACTOR = 1.5       # split input edges longer than this x the target edge length
SMOOTH_PASSES = 1


def _unit_normal(v, tri):
    n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
    area = np.linalg.norm(n)
    return n / max(area, 1e-15), area


def _opposite_angles(v, a, b, c):
    """Angle at vertex c of each triangle (a, b, c), for arrays of vertex indices."""
    x, y = v[a] - v[c], v[b] - v[c]
    return np.arctan2(np.linalg.norm(np.cross(x, y), axis=1), np.sum(x * y, axis=1))


def delaunay_flips(v, f, passes=20):
    """Flip edges whose two opposite angles sum to more than 180 degrees, worst first. This
    removes needle triangles and folded slivers without moving or adding a vertex, so the
    shape and SpHarm.birth stay valid. A flip is refused if it would leave a triangle
    pointing further from the local surface than the ones it replaces. Returns the new faces."""
    f = np.array(f)
    corners = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    for _ in range(passes):
        # All half-edges a->b with their opposite vertex c and face, as flat arrays
        a, b, c = f[:, corners].transpose(2, 0, 1).reshape(3, -1)
        face = np.repeat(np.arange(len(f)), 3)
        order = np.lexsort((np.arange(len(a)), a * len(v) + b))
        keys = (a * len(v) + b)[order]
        # Each half-edge a->b with a < b, matched to its twin b->a
        last = order[np.r_[keys[1:] != keys[:-1], True]]
        first = np.sort(last[a[last] < b[last]])
        pos = np.searchsorted(keys, b[first] * len(v) + a[first], side="right") - 1
        twin = order[np.maximum(pos, 0)]
        has_twin = (pos >= 0) & (a[twin] == b[first]) & (b[twin] == a[first])
        first, twin = first[has_twin], twin[has_twin]
        angle_sum = _opposite_angles(v, a[first], b[first], c[first]) + _opposite_angles(v, a[first], b[first], c[twin])
        worst_first = np.argsort(-angle_sum, kind="stable")
        worst_first = worst_first[angle_sum[worst_first] > np.pi + 1e-6]
        low, high = np.minimum(a, b), np.maximum(a, b)
        edges = set(zip(low.tolist(), high.tolist()))
        candidates = [(a[first[r]], b[first[r]], c[first[r]], c[twin[r]], face[first[r]], face[twin[r]]) for r in worst_first]
        touched, n_flips = set(), 0
        for a, b, c, d, i, k in candidates:
            a, b, c, d = int(a), int(b), int(c), int(d)
            if i in touched or k in touched or tuple(sorted((c, d))) in edges:
                continue
            n1, a1 = _unit_normal(v, (a, b, c))
            n2, a2 = _unit_normal(v, (b, a, d))
            m1, b1 = _unit_normal(v, (a, d, c))
            m2, b2 = _unit_normal(v, (d, b, c))
            mean = n1 * a1 + n2 * a2
            mean /= max(np.linalg.norm(mean), 1e-15)
            if min(b1, b2) < 1e-12 or min(m1 @ mean, m2 @ mean) < min(n1 @ mean, n2 @ mean) - 1e-9:
                continue
            f[i], f[k] = (a, d, c), (d, b, c)
            touched |= {i, k}
            edges.discard(tuple(sorted((a, b))))
            edges.add(tuple(sorted((c, d))))
            n_flips += 1
        if n_flips == 0:
            break
    return f


def split_long_edges(v, f, limit, max_passes=12):
    """Longest-edge bisection of every triangle whose longest edge exceeds limit; the mesh stays
    conforming and the shape is unchanged. Gives decimation small enough triangles to work with
    where the input mesh has large ones."""
    v, f = v.copy(), f.copy()
    for _ in range(max_passes):
        lengths = np.stack([np.linalg.norm(v[f[:, (i + 1) % 3]] - v[f[:, i]], axis=1) for i in range(3)], axis=1)
        longest = lengths.argmax(1)
        todo = np.where(lengths.max(1) > limit)[0]
        if not len(todo):
            break
        mid, verts = {}, list(v)
        for t in todo:
            a, b = f[t, longest[t]], f[t, (longest[t] + 1) % 3]
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(verts)
                verts.append((v[a] + v[b]) / 2)
        faces = []
        for tri in f:
            m = [mid.get((min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3]))) for i in range(3)]
            n = sum(x is not None for x in m)
            if n == 0:
                faces.append(tri)
            elif n == 3:
                faces += [[tri[0], m[0], m[2]], [tri[1], m[1], m[0]], [tri[2], m[2], m[1]], [m[0], m[1], m[2]]]
            elif n == 1:
                i = next(j for j in range(3) if m[j] is not None)
                faces += [[tri[i], m[i], tri[(i + 2) % 3]], [m[i], tri[(i + 1) % 3], tri[(i + 2) % 3]]]
            else:
                i = next(j for j in range(3) if m[j] is None)
                a, b, c = tri[(i + 1) % 3], tri[(i + 2) % 3], tri[i]
                x, y = m[(i + 1) % 3], m[(i + 2) % 3]
                faces += [[b, y, x], [c, x, y], [c, a, x]]
        v, f = np.array(verts), np.array(faces)
    return v, f


def smooth_on_surface(u, g, v, f, passes=SMOOTH_PASSES):
    """Evens out the decimated mesh (u, g): each pass moves vertices halfway to the mean of their
    neighbours along the surface, projects them back onto the input mesh (v, f), and flips edges."""
    for _ in range(passes):
        e = igl.edges(g)
        total, count = np.zeros_like(u), np.zeros(len(u))
        np.add.at(total, e[:, 0], u[e[:, 1]])
        np.add.at(total, e[:, 1], u[e[:, 0]])
        np.add.at(count, e[:, 0], 1)
        np.add.at(count, e[:, 1], 1)
        move = total / count[:, None] - u
        n = igl.per_vertex_normals(u, g)
        move -= np.sum(move * n, axis=1, keepdims=True) * n
        _, _, u = igl.point_mesh_squared_distance(u + 0.5 * move, v, f)
        g = delaunay_flips(u, g)
    return u, g


class SpHarm:
    """A decimated organoid mesh, its area-preserving map to the unit sphere, and the SPHARM
    expansions (shape reconstruction error, field power spectra) on that map."""

    def __init__(self):
        self.v = None  # Vertices
        self.f = None  # Faces
        self.coors = None  # Sphere coordinates (theta, phi) of each vertex
        self.transform_matrix = None  # PCA axes
        self._lw = None  # Cached lat, lon and quadrature weights

    def load_from_arrays(self, vertices, faces):
        self.v = vertices
        self.f = faces
        return self

    @classmethod
    def from_mesh(cls, v, f, n_faces=N_FACES):
        """Decimated copy of a mesh (true geometry, no rescaling) for the spherical map.

        Long input edges are split, the mesh is decimated, its needle triangles and folded
        slivers are removed by edge flips, and it is smoothed along the input surface.
        self.birth[i] is the index in `v` of the input vertex nearest decimated vertex i, so a
        per-vertex field on the full mesh is sampled as field[self.birth]. igl.decimate
        occasionally cannot reach the exact target on a nearly non-manifold mesh, so nearby
        targets are tried before giving up. A mesh already at or below n_faces is used as is."""
        sh = cls()
        if len(f) <= n_faces:
            sh.load_from_arrays(np.array(v, dtype=np.float64), np.array(f))
            sh.birth = np.arange(len(v))
            return sh
        area = igl.doublearea(v, f).sum() / 2
        vs, fs = split_long_edges(v, f, SPLIT_FACTOR * np.sqrt(4 * area / (np.sqrt(3) * n_faces)))
        for target in (n_faces, int(n_faces * 1.04), int(n_faces * 0.96), int(n_faces * 1.2)):
            ok, U, G, _, _ = igl.decimate(vs, fs, target)
            if ok:
                U = np.array(U, dtype=np.float64)
                U, G = smooth_on_surface(U, delaunay_flips(U, G), v, f)
                sh.load_from_arrays(U, G)
                sh.birth = cKDTree(v).query(U)[1]
                return sh
        raise ValueError(f"igl.decimate could not reach ~{n_faces} faces")

    def map_to_sphere(self, primal_steps=PRIMAL_STEPS):
        """PCA-align, start from the radial projection when it has no folds, else from the harmonic
        map (poles at the extremes of the first principal axis), then the area-preserving
        optimisation. Fills self.coors (theta, phi), self.start and self.rounds."""
        self.align_with_pca()
        p = self.v - self.v.mean(axis=0)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        volume = np.einsum('ij,ij->i', p[self.f[:, 0]], np.cross(p[self.f[:, 1]], p[self.f[:, 2]]))
        if min((volume < 0).sum(), (volume > 0).sum()) == 0:
            self.start = "radial"
            _, theta, phi = utils.cart2sph(*p.T)
            self.coors = np.vstack([theta, phi]).T
        else:
            self.start = "harmonic"
            self.compute_initial_parameterization()
        return self.optimize(max_outer_iterations=OUTER_ROUNDS, primal_steps=primal_steps)

    def sphere_points(self):
        x, y, z = utils.sph2cart(1, self.coors[:, 0], self.coors[:, 1])
        return np.stack([x, y, z], axis=1)

    def area_error(self):
        """Median relative error of the mapped triangle areas against the mesh triangle
        areas (both normalised to total 4 pi): how far the map is from area-preserving."""
        f = np.asarray(self.f)
        a_map = igl.doublearea(self.sphere_points(), f) / 2
        a_tgt = igl.doublearea(self.v, f) / 2
        a_tgt *= 4 * np.pi / a_tgt.sum()
        return float(np.median(np.abs(a_map - a_tgt) / a_tgt))

    def _latlon_weights(self):
        """Vertex lat/lon and quadrature weights (barycentric vertex areas on the mapped
        sphere), so the least-squares fit approximates the L2 projection on the sphere
        even where decimation left the vertices unevenly spaced."""
        if self._lw is None:
            lat, lon = utils.sph2latlon(self.coors[:, 0], self.coors[:, 1])
            w = igl.massmatrix(self.sphere_points(), self.f,
                               igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
            self._lw = lat, lon, w / w.mean()
        return self._lw

    def expand(self, field, lmax):
        """Weighted least-squares SH expansion (pyshtools SHCoeffs) of one per-vertex field."""
        lat, lon, w = self._latlon_weights()
        return pysh.SHCoeffs.from_least_squares(field, lat, lon, lmax, weights=w)

    def power_spectrum(self, fields, lmax):
        """Power per degree l = 0..lmax of each column of `fields` (n_verts, W): (lmax+1, W)."""
        fields = np.asarray(fields)
        return np.array([self.expand(fields[:, j], lmax).spectrum()
                         for j in range(fields.shape[1])]).T

    def complexity_errors(self, lmax_max):
        """Shape reconstruction error at lmax = 0..lmax_max: column l is the area-weighted
        RMS distance between the vertices and their degree-<=l SPHARM reconstruction (x, y,
        z each fitted at that lmax, evaluated at the vertices' sphere positions), divided
        by sqrt(surface area) so it is scale-free."""
        lat, lon, w = self._latlon_weights()
        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI).diagonal()
        errors = []
        for l in range(lmax_max + 1):
            recon = np.stack([self.expand(self.v[:, i], l).expand(lat=lat, lon=lon)
                              for i in range(3)], axis=1)
            d2 = np.sum((self.v - recon) ** 2, axis=1)
            errors.append(np.sqrt(np.sum(M * d2) / M.sum()) / np.sqrt(M.sum()))
        return np.array(errors)

    def save_map(self, path, **meta):
        np.savez(path, v=self.v, f=self.f, coors=self.coors, birth=self.birth,
                 transform_matrix=self.transform_matrix, **meta)

    @classmethod
    def load_map(cls, path):
        """(SpHarm, meta dict) from save_map's npz."""
        d = np.load(path, allow_pickle=True)
        sh = cls().load_from_arrays(d["v"], d["f"])
        sh.coors, sh.birth, sh.transform_matrix = d["coors"], d["birth"], d["transform_matrix"]
        return sh, {k: d[k][()] for k in d.files
                    if k not in ("v", "f", "coors", "birth", "transform_matrix")}

    def align_with_pca(self):
        """Rotate the mesh onto its principal axes."""
        pca = PCA(n_components=3)
        pca.fit(self.v)
        self.v = pca.transform(self.v)
        self.transform_matrix = np.copy(pca.components_)
        return self
    
    def compute_initial_parameterization(self):
        """Harmonic map to the sphere, the start when the radial projection has folds: theta solves
        Laplace's equation between the poles, phi solves it across a cut from pole to pole."""
        # Find extreme points as poles
        poles = np.array([np.argmax(self.v[:, 0]), np.argmin(self.v[:, 0])])
        pole_vals = np.array([0, np.pi])

        # List of all vertex indices and interior indices
        v_all = np.arange(self.v.shape[0])
        v_in = np.setdiff1d(v_all, poles)

        # Construct and slice up Laplacian
        l = igl.cotmatrix(self.v, self.f)
        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        Minv = sp.diags(1 / M.diagonal()) 
        l = Minv.dot(l)

        l_ii = l[v_in, :]
        l_ii = l_ii[:, v_in]

        l_ib = l[v_in, :]
        l_ib = l_ib[:, poles]

        # Solve PDE for theta
        theta_in = spsolve(-l_ii, l_ib.dot(pole_vals))

        theta = np.empty_like(self.v[:, 1])
        theta[poles] = pole_vals 
        theta[v_in] = theta_in
        
        # Compute branch cuts and phi
        phi = self._compute_phi(theta, poles, v_in)
        
        # Store the coordinates
        self.coors = np.vstack([theta, phi]).T
        return self
    
    def _find_rhs(self, displacements, forward_vec, backward_vec, n):
        """Helper function for computing branch cuts."""
        x_director = forward_vec/np.linalg.norm(forward_vec)
        y_director = np.cross(forward_vec, n)
        y_director /= np.linalg.norm(y_director)

        x_decom = np.dot(displacements, x_director)
        y_decom = np.dot(displacements, y_director)
        angle = np.arctan2(y_decom, x_decom) % (2*np.pi) 

        ref_x = np.dot(backward_vec, x_director)
        ref_y = np.dot(backward_vec, y_director)
        ref_angle = np.arctan2(ref_y, ref_x) % (2*np.pi) 

        return angle > (ref_angle + 1e-4)
    
    def _compute_phi(self, theta, poles, v_in):
        """Phi coordinates from Laplace's equation with a jump of 2 pi across the branch cut."""
        neighbours = igl.adjacency_list(self.f)
        normals = igl.per_vertex_normals(self.v, self.f)
        
        # Find branch cuts
        i = np.argmin(theta)
        line1 = []
        line2 = []
        
        while theta[i] < np.pi:
            neighbours_i = np.array(neighbours[i])
            n = normals[i]

            max_index = np.argmax(theta[neighbours_i])
            new_i = neighbours_i[max_index]
            neighbours_i = np.delete(neighbours_i, max_index)

            if len(line1) > 1:
                u = (self.v[new_i] - self.v[i])
                m = self._find_rhs(self.v[neighbours_i] - self.v[i], u, self.v[old_i]-self.v[i], n)
                line2.extend(neighbours_i[m])

            line1.append(new_i)
            old_i = i
            i = new_i
            
        line1 = np.array(line1, dtype=int)
        line2 = np.array(np.unique(line2), dtype=int)

        sort_indices = np.argsort(theta[line2])
        line2 = line2[sort_indices]
        
        # Solve PDE for phi
        l = igl.cotmatrix(self.v, self.f)

        for p in poles:
            for nn in neighbours[p]:
                l[nn, nn] += l[nn, p]  # delete all the connections to the poles

        M = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        Minv = sp.diags(1 / M.diagonal())
        l = Minv.dot(l)

        lines = np.concatenate([line1, line2])
        line_vals = np.concatenate([np.zeros(len(line1)), 2*np.pi*np.ones(len(line2))])
        v_in2 = np.setdiff1d(v_in, lines)  # cut off poles and branch cuts
        
        l_ii = l[v_in2, :]
        l_ii = l_ii[:, v_in2]

        l_ib = l[v_in2, :]
        l_ib = l_ib[:, lines]

        # Solve PDE
        phi_in = spsolve(-l_ii, l_ib.dot(line_vals))

        phi = np.empty_like(self.v[:, 1])
        phi[poles] = 0
        phi[v_in2] = phi_in
        phi[lines] = line_vals
        
        return phi
    
    @staticmethod
    def spherical_areas(u, f):
        """Signed spherical areas of triangles f on the unit vectors u (Van Oosterom & Strackee):
        a folded triangle has a negative area."""
        a, b, c = u[f[:, 0]], u[f[:, 1]], u[f[:, 2]]
        return 2 * jnp.arctan2(jnp.sum(a * jnp.cross(b, c), axis=1),
                               1 + jnp.sum(a * b, axis=1) + jnp.sum(b * c, axis=1) + jnp.sum(c * a, axis=1))

    def optimize(self, max_outer_iterations=OUTER_ROUNDS, primal_steps=PRIMAL_STEPS):
        """
        Run the augmented Lagrangian optimization method from the current self.coors.
        This finds a map of the mesh to the unit sphere that preserves the area around each vertex
        (a third of its triangles), minimising the Dirichlet energy of the map on the mesh, the
        sum over edges of cotangent weight (scaled to mean 1) x (1 - u_i . u_j). Each round is an
        L-BFGS solve; it stops once the energy has settled and the area constraints are met.

        Args:
            max_outer_iterations: Maximum number of outer iterations
            primal_steps: Maximum number of L-BFGS iterations per outer iteration

        Returns:
            self: For method chaining
        """
        y = self.sphere_points()
        f = np.asarray(self.f)

        # Wind the faces so that the initial map has positive total area
        if float(jnp.sum(self.spherical_areas(jnp.array(y), jnp.array(f)))) < 0:
            f = f[:, [0, 2, 1]]
        a_tgt = igl.doublearea(self.v, f) / 2
        a_tgt *= 4 * np.pi / a_tgt.sum()
        ring_tgt = np.zeros(len(y))
        np.add.at(ring_tgt, f.ravel(), np.repeat(a_tgt / 3, 3))

        edges = igl.edges(f)
        w = np.asarray(igl.cotmatrix(self.v, f).tocsr()[edges[:, 0], edges[:, 1]]).ravel()
        w, edges, faces, ring_tgt = jnp.array(w / w.mean()), jnp.array(edges), jnp.array(f), jnp.array(ring_tgt)

        def unit(y):
            return y / jnp.linalg.norm(y, axis=1, keepdims=True)

        def constraints(y):
            a = self.spherical_areas(unit(y), faces)
            return jnp.zeros(len(ring_tgt)).at[faces.ravel()].add(jnp.repeat(a / 3, 3)) / ring_tgt - 1

        def energy(y):
            u = unit(y)
            return jnp.sum(w * (1 - jnp.sum(u[edges[:, 0]] * u[edges[:, 1]], axis=1)))

        def loss(y, lam, mu):
            c = constraints(y)
            return energy(y) + jnp.sum(lam * c) + mu / 2 * jnp.sum(c * c)

        loss_and_grad, energy, constraints = jit(value_and_grad(loss)), jit(energy), jit(constraints)
        lam, mu, mu_max, mu_factor = jnp.zeros(len(ring_tgt)), 10.0, 1e3, 10.0
        prev = None

        for outer_iter in range(max_outer_iterations):
            def fun(z):
                value, g = loss_and_grad(jnp.array(z.reshape(-1, 3)), lam, mu)
                return float(value), np.array(g).ravel()

            res = minimize(fun, np.asarray(y).ravel(), jac=True, method="L-BFGS-B",
                           options=dict(maxiter=primal_steps, maxcor=20, ftol=1e-13, gtol=1e-7))
            y = unit(jnp.array(res.x.reshape(-1, 3)))
            c = constraints(y)
            lam = lam + mu * c
            cost = float(energy(y))
            self.rounds = outer_iter + 1
            if (prev is not None and abs(cost - prev) / cost < ENERGY_TOL
                    and float(jnp.abs(c).max()) < AREA_TOL):
                break
            prev = cost
            if outer_iter > 0 and outer_iter % 3 == 0:
                mu = min(mu_max, mu * mu_factor)

        _, theta, phi = utils.cart2sph(*np.array(y).T)
        self.coors = np.vstack([theta, phi]).T
        self._lw = None

        return self
