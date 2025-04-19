import torch
import torchdiffeq
import matplotlib.pyplot as plt
import numpy as np
import time
import dill

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Spatial_IFN_Model_2D(torch.nn.Module):
    def __init__(self,Nx,Ny,dx,dy,ps,mask):
        super().__init__()
        
        beta,phi,delta,k,f,p,this_pi,c,rho,DV,DF = ps
        
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy

        self.beta = beta
        self.phi = phi
        self.delta = delta
        self.k = k
        self.f = f
        self.p = p
        self.pi = this_pi
        self.c = c
        self.rho = rho

        self.DV = DV
        self.DF = DF

        self.mask = mask # Mask to handle boundaries for impenetrable regions.

    def laplacian_with_no_flux(self, C):
        C_up    = torch.roll(C, shifts=-1, dims=0)
        C_down  = torch.roll(C, shifts=1, dims=0)
        C_left  = torch.roll(C, shifts=1, dims=1)
        C_right = torch.roll(C, shifts=-1, dims=1)

        C_up[self.mask] = C[self.mask]
        C_down[self.mask] = C[self.mask]
        C_left[self.mask] = C[self.mask]
        C_right[self.mask] = C[self.mask]

        laplacian = ((C_right - 2 * C + C_left) / (self.dx ** 2) +
                     (C_up - 2 * C + C_down) / (self.dy ** 2))

        laplacian[self.mask] = 0
        return laplacian

    def forward(self,t,y):
        
        T, I, I_star, R, V, F = y[0], y[1], y[2], y[3], y[4], y[5]

        V_laplacian = self.laplacian_with_no_flux(V)
        F_laplacian = self.laplacian_with_no_flux(F)

        # Compute differentials
        dT_dt = -self.beta*V*T - self.phi*F*T + self.rho*R #dTdt([T, I, I_star, R, V, F],ps)
        dI_dt = self.beta*V*T - self.delta*I - self.k*I - self.phi*F*I #dIdt([T, I, I_star, R, V, F],ps)
        dI_star_dt = self.k*I + self.phi*F*I - self.delta*I_star #dIstardt([T, I, I_star, R, V, F],ps)
        dR_dt = self.phi*F*T - self.rho*R #dRdt([T, I, I_star, R, V, F],ps)
        dV_dt = self.DV * V_laplacian + self.p*I + (1.0-self.f)*self.p*I_star - self.c*V #DV*V_xx + dVdt([T, I, I_star, R, V, F],ps)
        dF_dt = self.DF * F_laplacian + self.pi*self.p*(I+I_star) - self.c*F #DF*F_xx + dFdt([T, I, I_star, R, V, F],ps)

        return torch.stack([dT_dt, dI_dt, dI_star_dt, dR_dt, dV_dt, dF_dt])

L = 10 # length is assumed to be 30cm: L = 30
Lx, Ly = L,L

domain = np.loadtxt("Intermediate-Objects/T0_map.txt")
print(domain)
print()
print(domain.shape)

### Other parameters, which are somewhat ambiguous in the paper.
RESOLUTION = 10
N = L*RESOLUTION # Spatial discretization
Nx = Lx*RESOLUTION
Ny = Ly*RESOLUTION

dx = Lx / (Nx - 1) # Grid spacing in x
dy = Ly / (Ny - 1) # Grid spacing in y

t0 = 0
tf = 16.0
#tf = 1.0

time_steps = int((tf-t0)*25)

### Primary system parameters

DV = 0.1 # virion diffusion
DF = 20*DV # IFN diffusion. Was 40

beta = 7.3e-2 # virus contact rate
phi = 7.3e-2 #IFN contact rate
delta = 4.0 #infected cell death rate
k = 2.0 # Autocrine transition rate
f = 0.9 # Autocrine efficiency
p = 250.0 # virion production # was 2400
this_pi = 0.5 # IFN production rate relative to virus production # Was 1.0
c = 14 # virion clearance

# From Table S2
rho = 1 # reversion from R to T

# Create spatial grid
x = torch.linspace(0, Lx, Nx, device=device)
y = torch.linspace(0, Ly, Ny, device=device)
X, Y = torch.meshgrid(x, y, indexing="ij")

x0 = 0*X*Y

T_0 = torch.zeros_like(x0)
I_0 = torch.zeros_like(x0)
I_star_0 = torch.zeros_like(x0)
R_0 = torch.zeros_like(x0)
V_0 = torch.zeros_like(x0)
F_0 = torch.zeros_like(x0)

T_0 = 100*domain # For now, we need to scale the domain values.
mask = T_0 == 0.0

T_0 = torch.tensor(T_0,device=device)

ps = [beta,phi,delta,k,f,p,this_pi,c,rho,DV,DF]
ps_reduced_delta = [beta,phi,0.5*delta,k,f,p,this_pi,c,rho,DV,DF]
ps_no_IFN = [beta,phi,0.5*delta,k,f,p,0.0,c,rho,DV,DF]

# Convert mask to tensor
mask = torch.tensor(mask, dtype=torch.bool)

NUM_POINTS = 5
initial_points_x = np.random.choice(Nx,NUM_POINTS,replace=False)
initial_points_y = np.random.choice(Ny,NUM_POINTS,replace=False)

initial_points = list(zip(initial_points_x,initial_points_y))

for point in initial_points:
    I_0[point] = 1.0

y0 = torch.stack([T_0,I_0,I_star_0,R_0,V_0,F_0]).to(device)

system = Spatial_IFN_Model_2D(Nx,Ny,dx,dy,ps,mask).to(device)
t_eval = torch.linspace(t0,tf,time_steps,device=device)

start = time.time()
solution = torchdiffeq.odeint(system, y0, t_eval, method='dopri5')
end = time.time()
print()
print(f"Time taken: {end - start}")

with open("Intermediate-Objects/solution.pkl","wb") as f:
    dill.dump(solution,f)
