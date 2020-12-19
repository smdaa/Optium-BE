# Fonction f1
function f1(A) 
    x = A[1];
    y = A[2];
    z = A[3];
    return 2 * (x + y + z - 3) ^ 2 + (x - y)^2 + (y - z)^2;
end
function grad_f1(A)
    x = A[1];
    y = A[2];
    z = A[3];   
    return  [4 * (x + y + z - 3) + 2 * (x - y); 4 * (x + y + z - 3) + 2 * (y - z) - 2 * (x -y); 4 * (x + y + z - 3) - 2 * (y - z)]
end
hess_f1(A) = [6 2 4; 2 8 2; 4 2 6]

# solution exacte
sol_exacte1 = [1; 1; 1]
sol_exactec1 = [0.5 1.25 0.5]

# points de depart
x011 = [1; 0; 0]
x012 = [10; 3; -2.2]
xc11 = [0; 1; 1]
xc12 = [0.5; 1.25; 1]

# Fonction f2
f2(x) = 100 * (x[2] - x[1] ^ 2) ^ 2 + (1 - x[1]) ^ 2
grad_f2(x) = [-400 * x[1] * (x[2] - x[1] ^ 2) - 2 * (1 - x[1]) ; 200 * (x[2] - x[1] ^ 2)]
hess_f2(x) = [-400 * (x[2] - 3 * x[1] ^ 2) + 2  -400 * x[1]; -400 * x[1]  200]

# solution exacte
sol_exacte2 = [1; 1]
sol_exactec2 = [0.9; 0.8]

# points de depart
x021 = [-1.2; 1]
x022 = [10; 0]
x023 = [0; ((1/200) + (1/(10^12)))]
xc21 = [1; 0]
xc22 = [sqrt(3)/2; sqrt(3)/2]

# Quadratique 1
g1 = [0; 0]
H1 = [7 0; 0 2]

# Quadratique 2
g2 = [6; 2]
H2 = [7 0; 0 2]

# Quadratique 3
g3 = [-2; 1]
H3 = [-2 0; 0 10]

# Quadratique 4
g4 = [0; 0]
H4 = [-2 0; 0 10]

# Quadratique 5
g5 = [2; 3]
H5 = [4 6; 6 5]

# Quadratique 6
g6 = [2; 0]
H6 = [4 0; 0 -15]

# fonction contrainte c1
c1(x) = x[1] + x[3] - 1
grad_c1(x) = [1; 0; 1]
hess_c1(x) = zeros(3,3)

# fonction contrainte c1
c2(x) = x[1] ^ 2 + x[2] ^ 2 - 1.5
grad_c2(x) = [2 * x[1]; 2 * x[2]]
hess_c2(x) = [2 0; 0 2];