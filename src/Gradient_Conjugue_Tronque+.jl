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
        max_iter = 1000
        tol = 1e-6
    else
        deltak = options[1]
        max_iter = options[2]
        tol = options[3]
    end

   "#pj est le vecteur de direction"
   n = length(gradfk)
   pj = -gradfk
   sj = zeros(n)
   gj = gradfk
   iter = 0
   s = zeros(n)

   while  iter <= max_iter
       println("daz")
        kappa_j = (pj') * hessfk * pj

        if kappa_j <= 0

            "# on écrit l'équation ||sj +x*pj|| = delta sous forme a*x^2 + b*x + c = 0 avec :"
            a = norm(pj)^2
            b = 2 * (sj') * pj
            c = norm(sj)^2 - deltak^2
            sqrt_determinant = sqrt(b^2 -4 * a * c)

            "# les racines de l'équation sont"
            racine1 = (- b - sqrt_determinant) / (2 * a)
            racine2 = (- b + sqrt_determinant) / (2 * a)

            "# calcul de q(sj + racine1*pj)"
            q_racine1 = (gj')*(sj + racine1*pj) +(1 / 2) * ((sj + racine1*pj)') * hessfk * (sj + racine1 * pj)
            "# calcul de q(sj + racine2*pj)"
            q_racine2 = (gj')*(sj + racine2*pj) +(1 / 2) * ((sj + racine2*pj)') * hessfk * (sj + racine2 * pj)

            "# on garde le s pour lequel la valeur de q est la plus petite"
            if q_racine1 < q_racine2
                sigma = racine1
            else
                sigma = racine2
            end
            s = sj + sigma * pj
            break
        end
        println(s)

       alphaj = norm(gj,2)^2 / kappa_j
       if norm(sj + alphaj * pj,2) >= deltak

            "# sigmaj est la racine positive de l’equation ‖sj+σpj‖ = ∆k"
            sigmaj = - norm(sj,2) + deltak / norm(pj,2)
            s = sj + sigmaj * pj
            break
       end

       "# Mise à jour des paramétres"
       sj = sj + alphaj*pj
       gjplus1 = gj + alphaj * hessfk * pj
       betaj = (norm(gjplus1,2) / norm(gj,2))^2
       pj = -gjplus1 + betaj * pj
       gj = gjplus1
       if (norm(gj,2)<tol*norm((gradfk),2))
            s = sj
            break
       end
       iter = iter + 1
   end
   return s
end
