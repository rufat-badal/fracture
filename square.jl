using JuMP
# using Ipopt
using MadNLP
using ProgressMeter
using CairoMakie

num_triangs_hor = 4 # choose even
num_triangs_ver = 4 # choose even
num_triangs = 2*num_triangs_hor * num_triangs_ver
num_edges = num_triangs_hor * (num_triangs_ver + 1) + num_triangs_ver * (2*num_triangs_hor + 1)
num_free_verts_hor = num_triangs_hor - 1 # ignore leftmost and rightmost vertices
num_free_verts_ver = num_triangs_ver + 1
num_verts_hor = num_free_verts_hor + 2
num_verts_ver = num_free_verts_ver
fps = 10
animation_fps = 60
animation_scale = 4
triang_side_length = 1
triang_height = sqrt(3)/2 * triang_side_length
pref_dist = triang_side_length
break_dist = 2*pref_dist
start_width = num_triangs_hor*triang_side_length
max_pull = 1.6*break_dist - triang_side_length
min_x = 0
max_x = start_width + max_pull
min_y = -triang_height
max_y = (num_triangs_ver + 1)*triang_height
pot_min = -1.
diss_coeff = 1.
time_pull = 2.
l2_dissipation = false # if false => Kelvin-Voigt
soft_max_alpha = 5
time_horizon = 2*time_pull
animation_width = 800
animation_aspect = (max_x - min_x) / (max_y - min_y)
animation_height = animation_width / animation_aspect
hidpi_scaling = 2
file_name = "square"
fontsize = 16
only_video = true; # if false generate a folder of snapshots for each frame

function lennard_jones(dist_sq; pot_min=-1)
    # Lennard-jones of squared distances.
    # The factor cbrt(2) assures that the global minimum 
    # is attained at cur_dist == pref_dist
    q = pref_dist^2/(cbrt(2)*dist_sq)
    -4*pot_min*(q^6 - q^3)
    # alternative power
    # q = pref_dist^2/(2*dist_sq)
    # -4*pot_min*(q^2 - q)
end

function dirichlet(t)
    start_width + max_pull*sin(pi/2*t/time_pull)
end

minmove = Model(
    ()->MadNLP.Optimizer(
        print_level=MadNLP.WARN,
        blas_num_threads=8,
        # acceptable_tol=1e-8,
        # max_iter=1000
    )
)

# minmove = Model(Ipopt.Optimizer)
# set_optimizer_attribute(minmove, "max_cpu_time", 60.0)
# set_optimizer_attribute(minmove, "print_level", 0)

step = 2
prev_L = num_triangs_hor*triang_side_length
L = dirichlet((step-1)/fps)

@NLparameter(
    minmove,
    prev_x[i=1:num_free_verts_hor,j=1:num_free_verts_ver] ==
        i*triang_side_length + ((j+1) % 2)*triang_side_length/2
)
@NLparameter(
    minmove,
    prev_y[i=1:num_free_verts_hor,j=1:num_free_verts_ver] ==
        (j-1)*triang_height
)

delta_L = L - prev_L
# apply uniform correction
@variable(
    minmove,
    min_x <= x[i=1:num_free_verts_hor,j=1:num_free_verts_ver] <= max_x,
    start = value(prev_free_x[i,j]) + delta_L/num_free_verts_hor
)
@variable(
    minmove,
    min_y <= y[i=1:num_free_verts_hor,j=1:num_free_verts_ver] <= max_y,
    start = value(prev_free_x[i,j]) + delta_L/num_free_verts_hor
)

# Dirichlet condition
@NLparameter(
    minmove,
    prev_bdry_x[j=1:num_free_verts_ver] == prev_L + ((j+1) % 2)*triang_side_length/2
)
@NLparameter(
    minmove,
    bdry_x[j=1:num_free_verts_ver] == L + ((j+1) % 2)*triang_side_length/2
)

# vertices 
prev_vertices = Matrix{Any}(undef, num_verts_hor, num_verts_ver)
vertices = Matrix{Any}(undef, num_verts_hor, num_verts_ver)
# leftmost (fixed)
for j in 1:num_verts_ver
    prev_vertices[1,j] = [
        ((j+1) % 2)*triang_side_length/2,
        (j-1)*triang_height
    ]
    vertices[1,j] = prev_vertices[1,j]
end
# middle (free)
for i in 1:num_free_verts_hor
    for j in 1:num_free_verts_ver
        prev_vertices[i+1,j] = [
            prev_x[i,j],
            prev_y[i,j]
        ]
        vertices[i+1,j] = [
            x[i,j],
            y[i,j]
        ]
    end
end
# rightmost (driven by the boundary condition)
for j in 1:num_verts_ver
    prev_vertices[num_verts_hor, j] = [
        prev_bdry_x[j],
        (j-1)*triang_height
    ]
    vertices[num_verts_hor, j] = [
        bdry_x[j],
        (j-1)*triang_height
    ]
end

function get_edges(verts)
    edges = Matrix{Vector{Any}}(undef, num_edges, 2)
    edge_id = 1
    # add horizontal edges
    for i in 1:num_verts_hor-1
        for j in 1:num_verts_ver
            edges[edge_id, 1] = verts[i,j]
            edges[edge_id, 2] = verts[i+1,j]
            edge_id += 1
        end
    end
    # right-leaning odd rows
    for i in 1:num_verts_hor
        for j in 1:2:num_verts_ver-1
            edges[edge_id, 1] = verts[i,j]
            edges[edge_id, 2] = verts[i,j+1]
            edge_id += 1
        end
    end
    # right-leaning even rows
    for i in 1:num_verts_hor-1
        for j in 2:2:num_verts_ver-1
            edges[edge_id, 1] = verts[i,j]
            edges[edge_id, 2] = verts[i+1,j+1]
            edge_id += 1
        end
    end
    # left-leaning odd rows
    for i in 2:num_verts_hor
        for j in 1:2:num_verts_ver-1
            edges[edge_id, 1] = verts[i,j]
            edges[edge_id, 2] = verts[i-1,j+1]
            edge_id += 1
        end
    end
    # left-leaning even rows
    for i in 1:num_verts_hor
        for j in 2:2:num_verts_ver-1
            edges[edge_id, 1] = verts[i,j]
            edges[edge_id, 2] = verts[i,j+1]
            edge_id += 1
        end
    end 
    edges
end

prev_edges = get_edges(prev_vertices)
edges = get_edges(vertices)

function get_triangles(verts)
    triangs = Matrix{Vector{Any}}(undef, num_triangs, 3)
    triang_id = 1
    # upward-pointing odd rows
    for i in 1:num_verts_hor-1
        for j in 1:2:num_verts_ver-1
            triangs[triang_id, 1] = verts[i,j]
            triangs[triang_id, 2] = verts[i+1,j]
            triangs[triang_id, 3] = verts[i,j+1]
            triang_id += 1
        end
    end
    # upward-pointing even rows
    for i in 1:num_verts_hor-1
        for j in 2:2:num_verts_ver-1
            triangs[triang_id, 1] = verts[i,j]
            triangs[triang_id, 2] = verts[i+1,j]
            triangs[triang_id, 3] = verts[i+1,j+1]
            triang_id += 1
        end
    end
    # downward-pointing odd rows
    for i in 1:num_verts_hor-1
        for j in 3:2:num_verts_ver
            triangs[triang_id, 1] = verts[i,j]
            triangs[triang_id, 2] = verts[i,j-1]
            triangs[triang_id, 3] = verts[i+1,j]
            triang_id += 1
        end
    end
    # downward-pointing odd rows
    for i in 1:num_verts_hor-1
        for j in 2:2:num_verts_ver
            triangs[triang_id, 1] = verts[i,j]
            triangs[triang_id, 2] = verts[i+1,j-1]
            triangs[triang_id, 3] = verts[i+1,j]
            triang_id += 1
        end
    end
    triangs
end

prev_triangles = get_triangles(prev_vertices)
triangles = get_triangles(vertices)

function vector_to_values(vec)
    broadcast((p) -> value.(p), vec)
end

function plot_grid(vertices, edges, triangles)
    fig = Figure(
        resolution=(
            animation_scale*animation_width,
            animation_scale*animation_height
        ), fontsize=animation_scale*fontsize
    )
    ax = Axis(
        fig[1, 1],
        limits=(min_x-0.25, max_x, min_y, max_y),
        aspect=animation_aspect,
    )

    plot_triangles!(ax, vector_to_values(triangles))
    plot_edges!(ax, vector_to_values(edges))
    plot_vertices!(ax, vector_to_values(vertices))

    fig
end

function plot_vertices!(ax, vertices)
    vertices = vertices[:]
    x = [v[1] for v in vertices]
    y = [v[2] for v in vertices]
    scatter!(ax, x, y, color=:blue, markersize=animation_scale*10)
end

function plot_edges!(ax, edges)
    num_edges = size(edges)[1]
    for i in 1:num_edges
        v1, v2 = edges[i, :]
        x1, y1 = v1
        x2, y2 = v2
        lines!(
            ax, [x1, x2], [y1, y2],
            color=:green,
            linewidth=animation_scale*3,
            linestyle=:dash
        )
    end
end

function plot_triangles!(ax, triangles)
    num_triags = size(triangles)[1]

    for i in 1:num_triags
        poly!(ax, Tuple.(triangles[i, :]), color=(:pink, 0.5))
    end
end

plot_grid(prev_vertices, prev_edges, prev_triangles)