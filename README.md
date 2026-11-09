> 🌐 **Démo Interactive Déployée en Ligne :** [**https://collesyll0607.github.io/prediction-loyers-ml/**](https://collesyll0607.github.io/prediction-loyers-ml/)
>
> 👩‍💻 **Portfolio de l'Auteure (Collé Syll) :** [https://collesyll0607.github.io/](https://collesyll0607.github.io/)

---

# Prédiction des Prix de Location Immobilière — Machine Learning (UASZ AI Community)

**Auteure** : Collé Syll — Licence 3 Ingénierie Informatique, Université Assane Seck de Ziguinchor (UASZ)  
**Cadre** : Projet collaboratif Machine Learning — **UASZ AI Community**  
**Technologies** : `Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Random Forest`

## Présentation
Développement collaboratif au sein de l'**UASZ AI Community** d'un pipeline de Machine Learning supervisé permettant de prédire les loyers mensuels immobiliers à partir des caractéristiques des logements (`surface_m2`, `nb_pieces`, `nb_salles_bain`, `quartier`, `meuble`, `neuf_ou_renove`, `distance_transport_km`).

Le projet couvre l'ensemble du cycle de vie Data Science :
1. **Traitement et nettoyage des données** avec `Pandas` et `NumPy`.
2. **Entraînement supervisé** d'un modèle ensembliste **Random Forest Regressor** avec `Scikit-learn`.
3. **Évaluation statistique** : coefficient de détermination **$R^2 \approx 0,77$** sur le jeu de test.

## Exécution rapide

```bash
# 1. Entraîner le modèle et générer les métriques
python generate_and_train.py

# 2. Exécuter les tests unitaires
python -m unittest test_pipeline_ml.py -v
```

