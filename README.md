# Prédiction des Prix de Location Immobilière — Machine Learning

**Auteure** : Collé Syll — Licence 3 Ingénierie Informatique (UASZ)  
**Technologies** : `Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Random Forest`

## Présentation
Développement d'un pipeline de Machine Learning supervisé permettant de prédire les loyers mensuels immobiliers à partir des caractéristiques des logements (`surface_m2`, `nb_pieces`, `nb_salles_bain`, `quartier`, `meuble`, `neuf_ou_renove`, `distance_transport_km`).

Le modèle **Random Forest Regressor** entraîné et évalué sur l'échantillon de test atteint un coefficient de détermination **$R^2 \approx 0,77$**.

## Exécution rapide

```bash
# 1. Entraîner le modèle et générer les métriques
python generate_and_train.py

# 2. Exécuter les tests unitaires
python -m unittest test_pipeline_ml.py -v
```
