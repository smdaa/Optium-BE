@doc doc"""
Résolution des problèmes de minimisation sous contraintes d'égalités

# Syntaxe
```julia
Lagrangien_Augmente(algo,fonc,contrainte,gradfonc,hessfonc,grad_contrainte,
			hess_contrainte,x0,options)
```

# Entrées
  * **algo** 		   : (String) l'algorithme sans contraintes à utiliser:
    - **"newton"**  : pour l'algorithme de Newton
    - **"cauchy"**  : pour le pas de Cauchy
    - **"gct"**     : pour le gradient conjugué tronqué
  * **fonc** 		   : (Function) la fonction à minimiser
  * **contrainte**	   : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  * **gradfonc**       : (Function) le gradient de la fonction
  * **hessfonc** 	   : (Function) la hessienne de la fonction
  * **grad_contrainte** : (Function) le gradient de la contrainte
  * **hess_contrainte** : (Function) la hessienne de la contrainte
  * **x0** 			   : (Array{Float,1}) la première composante du point de départ du Lagrangien
  * **options**		   : (Array{Float,1})
    1. **epsilon** 	   : utilisé dans les critères d'arrêt
    2. **tol**         : la tolérance utilisée dans les critères d'arrêt
    3. **itermax** 	   : nombre maximal d'itération dans la boucle principale
    4. **lambda0**	   : la deuxième composante du point de départ du Lagrangien
    5. **mu0,tho** 	   : valeurs initiales des variables de l'algorithme

# Sorties
* **xmin**		   : (Array{Float,1}) une approximation de la solution du problème avec contraintes
* **fxmin** 	   : (Float) ``f(x_{min})``
* **flag**		   : (Integer) indicateur du déroulement de l'algorithme
   - **0**    : convergence
   - **1**    : nombre maximal d'itération atteint
   - **(-1)** : une erreur s'est produite
* **niters** 	   : (Integer) nombre d'itérations réalisées

# Exemple d'appel
```julia
using LinearAlgebra
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
algo = "gct" # ou newton|gct
x0 = [1; 0]
options = []
contrainte(x) =  (x[1]^2) + (x[2]^2) -1.5
grad_contrainte(x) = [2*x[1] ;2*x[2]]
hess_contrainte(x) = [2 0;0 2]
output = Lagrangien_Augmente(algo,f,contrainte,gradf,hessf,grad_contrainte,hess_contrainte,x0,options)
```
"""
function Lagrangien_Augmente(algo,fonc::Function,contrainte::Function,gradfonc::Function,
	hessfonc::Function,grad_contrainte::Function,hess_contrainte::Function,x0,options)

	if options == []
		epsilon = 1e-8
		tol = 1e-5
		itermax = 1000
		lambda0 = 2
		mu0 = 100
		tho = 2
	else
		epsilon = options[1]
		tol = options[2]
		itermax = options[3]
		lambda0 = options[4]
		mu0 = options[5]
		tho = options[6]
  end
  
  if (algo != "newton") && (algo != "cauchy") && (algo != "gct")
    flag = -1
    println("Usage : algo = newton | cauchy | gct")
    return
  end

  n = length(x0)
  xmin = zeros(n)
	fxmin = 0
	flag = 0
  iter = 0
  mu = mu0
  lambda = lambda0
  epsilon0 = 1 / mu0
  epsilon = epsilon0
  alpha = 0.1
  beta = 0.9
  etac0 = 0.1258925
  eta0 = etac0 / (mu0 ^ alpha)
  eta = eta0
  etac = etac0

  while (iter <= itermax)
    LA(x) = fonc(x) + transpose(lambda) * contrainte(x) + (1 / 2) * mu * (norm(contrainte(x)) ^ 2)
    grad_LA(x) = gradfonc(x) + transpose(lambda) * grad_contrainte(x) + mu * transpose(grad_contrainte(x)) * contrainte(x)
    hess_LA(x) = hessfonc(x) + transpose(lambda) * hess_contrainte(x) + mu * (transpose(grad_contrainte(x)) * grad_contrainte(x))
    #a
    if algo == "newton"
      xl, ~ = Algorithme_De_Newton(LA, grad_LA, hess_LA, x0, [])
    elseif algo == "cauchy"
      xl, ~ = Regions_De_Confiance("cauchy", LA, grad_LA, hess_LA, x0, [])
    elseif algo == "gct"
      xl, ~ = Regions_De_Confiance("gct", LA, grad_LA, hess_LA, x0, [])
    end
    convergence = (norm(grad_LA(xl)) <= tol * norm(grad_LA(xl))) && (norm(contrainte(xl)) <= tol * norm(contrainte(xl)))
    if convergence 
      xmin = xl
      break
    end
    #b
    if (norm(contrainte(xl)) <= eta)
      lambda = lambda + mu * contrainte(xl)
      epsilon = epsilon / mu
      eta = eta / (mu ^ beta)
    else
      mu = tho * mu
      epsilon = epsilon0 / mu
      eta = etac0 / (mu ^ alpha)
    end
    iter = iter + 1
  end
  if (iter > itermax)
    flag = 1
  end
	fxmin = fonc(xmin)
	return xmin,fxmin,flag,iter
end
