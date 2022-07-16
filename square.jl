using JuMP
using MadNLP
using ProgressMeter
using CairoMakie

num_triangs_hor = 4 # choose even
num_triangs_ver = 4 # choose even
num_triangs = (2*num_triangs_hor-1) * num_triangs_ver
num_edges = (num_triangs_hor - 1) * (num_triangs_ver + 1) +
    1 + num_triangs_ver ÷ 2 +
    num_triangs_hor * 2 * num_triangs_ver
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
min_y = -2*triang_height
max_y = (num_triangs_ver + 2)*triang_height
pot_min = -1.
diss_coeff = 1.
time_pull = 2.
l2_dissipation = false # if false => Kelvin-Voigt
soft_max_alpha = 5
time_horizon = 2*time_pull
animation_width = 800
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
        # acceptable_tol=1e-8,
        # max_iter=1000
    )
)

step = 2
prev_L = num_triangs_hor*triang_side_length
L = dirichlet((step-1)/fps)

@NLparameter(
    minmove,
    prev_free_x[i=1:num_free_verts_hor,j=1:num_free_verts_ver] ==
        (i-1)*triang_side_length + ((j+1) % 2)*triang_side_length/2
)
@NLparameter(
    minmove,
    prev_free_y[i=1:num_free_verts_hor,j=1:num_free_verts_ver] ==
        (j-1)*triang_height
)

delta_L = L - prev_L
@variable(
    minmove,
    min_x <= free_x[i=1:num_free_verts_hor,j=1:num_free_verts_ver] <= max_x,
    start = value(prev_free_x[i,j]) + delta_L/num_free_verts_hor
)
@variable(
    minmove,
    min_y <= free_y[i=1:num_free_verts_hor,j=1:num_free_verts_ver] <= max_y,
    start = value(prev_free_x[i,j]) + delta_L/num_free_verts_hor
)

@NLparameter(
    minmove,
    right_x[j=1:num_free_verts_ver] == L + ((j+1) % 2)*triang_side_length/2
)

# x- and y-coordinates of the whole grid as expressions
x = Matrix{Any}(undef, num_verts_hor, num_verts_ver)
y = Matrix{Any}(undef, num_verts_hor, num_verts_ver)
# leftmost (fixed)


# middle (free)

# rightmost (driven by the boundary condition)