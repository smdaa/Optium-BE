@doc doc"""
Minimise le problème : ``min_{||s||< \delta_{k}} q_k(s) = s^{t}g + (1/2)s^{t}Hs``
                        pour la ``k^{ème}`` itération de l'algorithme des régions de confiance

# Syntaxe
```julia
sk = Gradient_Conjugue_Tronque(fk,gradfk,hessfk,option)
```

# Entrées :
   * **gradfk**           : (Array{Float,1}) le gradient de la fonction f appliqué au point xk
   * **hessfk**           : (Array{Float,2}) la Hessienne de la fonction f appliqué au point xk
   * **options**          : (Array{Float,1})
      - **delta**    : le rayon de la région de confiance
      - **max_iter** : le nombre maximal d'iterations
      - **tol**      : la tolérance pour la condition d'arrêt sur le gradient


# Sorties:
   * **s** : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \delta_{k}} q(s)``

# Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(gradfk,hessfk,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        deltak = 2
        max_iter = 100
        tol = 1e-6
    else
        deltak = options[1]
        max_iter = options[2]
        tol = options[3]
    end

   n = length(gradfk)
   sj = zeros(n)
   s = zeros(n)
   p = -(gradfk)
   g = gradfk


   for j = 1:max_iter
        #a
        k = transpose(p) * hessfk * p

        #b
        if (k <= 0)
            discriminant = 4 * (transpose(sj) * p)^2 - 4 * (norm(p)^2) * ((norm(sj)^2) - deltak^2)

            sigma1 = (-2 * transpose(sj) * p - sqrt(discriminant)) / (2 * (norm(p)^2))
            sigma2 = (-2 * transpose(sj) * p + sqrt(discriminant)) / (2 * (norm(p)^2))

            s1 = sj + sigma1 * p
            s2 = sj + sigma2 * p


            s=min((transpose(gradfk) * s1 + (1 / 2) * transpose(s1) * hessfk * s1),(transpose(gradfk) * s2 + (1 / 2) * transpose(s2) * hessfk * s2))
            break
       end
       #c
       alpha = transpose(g) * g / k

       #d
       if norm(sj + alpha * p) >= deltak
           discriminant = 4 * (transpose(sj) * p)^2 - 4 * (norm(p)^2) * ((norm(sj)^2) - deltak^2)
           sigma1 = (-2 * transpose(sj) * p - sqrt(discriminant)) / (2 * (norm(p)^2))
           sigma2 = (-2 * transpose(sj) * p + sqrt(discriminant)) / (2 * (norm(p)^2))

           #On prend la veluer de sigma qui est positive
           s = sj + max(sigma1,sigma2) * p

           break
       end



       #e
       sj = sj + alpha*p

       #f
       g_next = g + alpha * hessfk * p

       #g
       beta = transpose(g_next)*g_next/(transpose(g)*g)

       #h
       p = -g_next + beta*p
       g = g_next

       #i
       convergence = (norm(g,2)<tol*norm((gradfk),2))

       if convergence
           s=sj
           break
       end
   end

return s

end
