

<h1>Segmentation d'Images pour Véhicules Autonomes</h1>

<div class='img'>
  <img src='https://images.unsplash.com/photo-1485463611174-f302f6a5c1c9?q=80&w=1752&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D', alt='car from inside'>
</div> 
<p>Conception d'un modèle de segmentation d'images embarqué pour la vision par ordinateur</p>


<p>Conception d'un modèle de segmentation d'images embarqué pour la vision par ordinateur chez Future Vision Transport.</p>

<h2>Introduction</h2>

<p>Future Vision Transport est à la pointe de la technologie des véhicules autonomes. Nous développons des systèmes de vision par ordinateur qui permettent aux véhicules de "voir" et de comprendre leur environnement. La segmentation d'images est un élément essentiel de cette technologie, car elle permet de distinguer les différents objets présents sur une image, comme les voitures, les piétons, les routes, etc.</p>

<h2>Objectif du Projet</h2>

<p>Notre objectif est de concevoir un modèle de segmentation d'images optimisé pour les systèmes embarqués de nos véhicules. Ce modèle doit être à la fois :</p>

Performant : Capable de détecter et de classifier avec précision les différents éléments de la scène.
Léger : Suffisamment compact pour fonctionner sur les ressources limitées des systèmes embarqués.
Facile à utiliser : Nous développerons une API simple pour que notre système de décision puisse facilement utiliser les résultats de la segmentation.
<p>Nous créerons également une application web pour visualiser les résultats de la segmentation. Cette application nous permettra de tester et d'améliorer notre modèle, ainsi que de communiquer nos résultats à l'ensemble de l'équipe.</p>

<h2>Contraintes et Besoins</h2>

<p>Notre projet doit répondre aux besoins spécifiques de nos collègues :</p>

<h3>Contraintes de Franck (Traitement d'Images)</h3>
<ul>
<li>Jeu de données Cityscapes :  Nous utiliserons ce jeu de données de référence pour entraîner notre modèle. Il contient des milliers d'images de scènes urbaines annotées avec 8 catégories principales (voiture, piéton, route, etc.).</li>
<li>Ressources limitées : Notre modèle doit être optimisé pour fonctionner sur les systèmes embarqués des véhicules, qui ont une puissance de calcul et une mémoire limitées.</li>
</ul>

<h3>Besoins de Laura (Système de Décision)</h3>
<ul>
<li>API simple : Notre API doit être facile à utiliser et à intégrer dans le système de décision.</li>
<li>Visualisation claire : L'application web doit permettre de visualiser les résultats de la segmentation de manière intuitive et compréhensible.</li>
</ul>

### Livrables
- Fiche technique : [PDF](https://github.com/Zaccaria-Amillou/ociap8/blob/main/Fiche_Technique.pdf)
- EDA : [Notebook](https://github.com/Zaccaria-Amillou/ociap8/blob/main/notebook/eda.ipynb)
- Modélisation : [Notebook](https://github.com/Zaccaria-Amillou/ociap8/blob/main/notebook/modelisation.ipynb)
- API : [Fichier](https://github.com/Zaccaria-Amillou/app_image)
