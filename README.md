
# Hardware for Signal Processing

TP réalisé par __Théo BARBEROT & Thomas BONNEVIALLE__

# Implémentation d'un CNN - Le Net5 sur GPU

## Objectifs et Méthodes

* Le but de ce TP est de programmer en langage CUDA un CNN pour la classification de nombres. Le CUDA est un langage informatique similaire au C permettant d'effectuer plusieurs actions identiques simultanément. Par exemple, pour calculer l'addition de 2 matrices, nous pouvons effectuer l'addition de chaque case *M[i,j]* en même temps.  
* Cela va nous permettre alors de comparer l'exécution et le temps de calcul d'un algorithme utilisant les *grid*, *bloc* et *thread* à un algorithme faisant les calculs naïvement. Mais également de remarquer les limites de l'utilisation d'un GPU.  
* Nous allons ensuite implémenter un CNN (seulement la partie inférence et non l'entrainement) et exporter les données d'un notebook python pour un projet cuda.  
* Le TP permet également d'améliorer notre connaissance sur l'outil git.  

Pour coder en CUDA, nous avons utilisé l'interface graphique (l'IDE) de VSCode : il suffit ensuite de se placer dans le dossier contenant notre fichier *.cu*. La compilation s'effectue alors via la commande :

`nvcc nomfichier.cu -o nomfichier`

L'exécution s'effectue quant à elle de la manière suivante :

`./nomfichier` ou `time./nomfichier` si nous voulons accéder au temps mis pour faire le calcul.


L'architecture de LeNet-5 est la suivante :
![image](https://user-images.githubusercontent.com/94001440/212563409-2570af1b-9190-4663-bf24-1ea083c13a3c.png)


## Prise en main de CUDA

En premier lieu, nous avons appris à coder en langage CUDA. Nous nous sommes donc intéressés à l'addition ainsi qu'à la multiplication de matrices. Nous avons codé ces 2 fonctions en utilisant le CPU et le GPU pour comparer la vitesse d'exécution des 2 programmes. Nous avons tout d'abord créé une fonction permettant d'initialiser une matrice et une autre permettant de la visualiser.

Voici un exemple d'une addition et d'une multiplication sur le CPU de matrices de taille 3x3. Les deux premières matrices, nommées respectivement M1 et M2, dans notre code sont les matrices initialisées entre 0 et 1. La troisième correspond à l'addition de ces deux matrices alors que la dernière représente la multiplication.  
![image](https://user-images.githubusercontent.com/94001440/212568280-fa235c62-510a-45fa-ab6c-b7bd29d72170.png)

Lorsque nous effectuons ces opérations sur le GPU, nous avons dû veiller à utiliser correctement les *grid*, *bloc* et *thread* pour gérer l'indice du tableau. Nous avons obtenu les mêmes résultats.

Pour simplifier les tâches et éviter les calculs du CPU quand nous sommes dans le GPU et inversement, nous avons modifié notre code pour mettre en entrée si nous utilisons le GPU ou le CPU ainsi que la taille des matrices que nous initialisons. Par exemple :
`./ gpu 3 3`

Comparons maintenant le temps de calcul de l'addition suivi de la multiplication sur le CPU puis sur le GPU.  
![image](https://user-images.githubusercontent.com/94001440/212568656-7b520572-2b5e-42a2-8f12-8b7c1758c3ec.png)  
Nous remarquons que pour des matrices de taille 1000x1000, le temps de calcul sur GPU est beaucoup plus faible que le temps de calcul sur le GPU (quelques millisecondes comparées aux dizaines de secondes).


## Premières couches du réseau de neurone LeNet-5

Nous avons maintenant programmé différentes couches du réseau LeNet-5 : la convolution, le sous-échantillonnage et une fonction d'activation.  
![image](https://user-images.githubusercontent.com/94001440/212569652-9f8eec2e-c039-4336-bb75-1805b2f05b08.png)  
Sur cette photo, les matrices correspondent respectivement à notre matrice initiale, au filtre de convolution, à la matrice résultant de la convolution et à la matrice correspondant à un sous échantillonnage d'ordre 2.  
Nous avons réussi à programmer ces différentes fonctions et à obtenir les résultats que nous souhaitions. Nous allons ensuite les assembler pour créer notre CNN.


## Un peu de python

Le réseau LeNet-5 est une succession de couches de convolutions, de fonctions d'activation ainsi que de sous-échantillonnage. Nous convoluons tout d'abord les images de tailles 32x32 par 6 noyaux de convolution de taille 5x5 et appliquons une fonction d'application. La matrice obtenue de taille 6x28x28 est alors sous-échantillonnée d'un facteur 2 pour obtenir une matrice de taille 6x14x14. Nous appliquons de nouveau les 6 noyaux de convolution ainsi que la fonction d'activation et le sous-échantillonnage d'un facteur 2 pour obtenir en sortie une matrice de taille 36x5x5. Nous devons ensuite rajouter une fonction *dense* permettant de classifier en multipliant la matrice de sortie par une fonction de densité de la forme *Wx+b*.  
Nous devons alors exporter les poids du notebook python que nous avons. Nous obtenons un fichier *.h5* que nous devons ouvrir avec la fonction FILE. Pour que le réseau soit opérationnel, nous devons finir cette partie.
