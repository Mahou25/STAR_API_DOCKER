import os
import json
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from PIL import Image
import tempfile
import atexit
import glob
import threading
import time
import threading
import time
from functools import wraps
import socket




def convertir_numpy_pour_json(obj):
    """
    Convertit récursivement les types NumPy en types Python natifs pour la sérialisation JSON
    """
    if isinstance(obj, dict):
        return {key: convertir_numpy_pour_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convertir_numpy_pour_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Importations pour la visualisation 3D (EXACTEMENT comme dans vos scripts)
try:
    from vedo import Mesh, Plotter, show, colors as vcolors, Points, screenshot
    VEDO_AVAILABLE = True
except ImportError:
    print("Vedo non disponible - visualisation 3D limitée")
    VEDO_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration des dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAR_DIR = os.path.join(BASE_DIR, 'star_1_1')
GENERATED_DIR = os.path.join(BASE_DIR, 'generated')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
PREVIEW_DIR = os.path.join(BASE_DIR, 'previews')

# Créer les dossiers nécessaires
for directory in [GENERATED_DIR, TEMP_DIR, PREVIEW_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- COULEURS DISPONIBLES (EXACTEMENT comme dans vos scripts) ---
COULEURS_DISPONIBLES = {
    "Noir": [25, 25, 25],
    "Bleu Marine": [25, 25, 128],
    "Gris Anthracite": [77, 77, 77],
    "Bordeaux": [128, 25, 51],
    "Rouge": [204, 25, 25],
    "Bleu Clair": [77, 102, 204],
    "Vert Olive": [102, 128, 51],
    "Marron": [102, 77, 51],
    "Blanc Cassé": [230, 230, 217],
    "Rose Poudré": [204, 153, 179],
}

# --- TYPES DE VÊTEMENTS ---
TYPES_VETEMENTS = {
    "Mini-jupe droite": {
        "categorie": "jupe",
        "type": "droite",
        "longueur_relative": 0.15,
        "description": "Mini-jupe droite (mi-cuisse)"
    },
    "Jupe droite au genou": {
        "categorie": "jupe",
        "type": "droite",
        "longueur_relative": 0.35,
        "description": "Jupe droite classique (genou)"
    },
    "Jupe droite longue": {
        "categorie": "jupe",
        "type": "droite",
        "longueur_relative": 0.75,
        "description": "Jupe droite longue (cheville)"
    },
    "Mini-jupe ovale": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.15,
        "ampleur": 1.5,           # ✅ CORRECTION: augmenté de 1.3 à 1.5
        "description": "Mini-jupe ovale évasée (mi-cuisse)"
    },
    "Jupe ovale au genou": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.35,
        "ampleur": 1.6,           # ✅ CORRECTION: augmenté de 1.4 à 1.6
        "description": "Jupe ovale classique (genou)"
    },
    "Jupe trapèze au genou": {
        "categorie": "jupe",
        "type": "trapeze",
        "longueur_relative": 0.35,
        "evasement": 1.6,
        "description": "Jupe trapèze classique (genou)"
    },
}

# --- MAPPING DES MESURES (comme dans le script 2) ---
DEFAULT_MAPPING = {
    "tour_poitrine": {"joints": [17, 18], "description": "Tour de poitrine"},
    "tour_taille": {"joints": [1], "description": "Tour de taille"},
    "tour_hanches": {"joints": [2], "description": "Tour de hanches"},
    "hauteur": {"joints": [0, 10], "description": "Hauteur totale"},
    "longueur_bras": {"joints": [16, 20], "description": "Longueur des bras"}
}

# ============== CLASSES EXACTES DE VOS SCRIPTS ==============

class MannequinGenerator:
    """EXACTEMENT la même logique que dans le script 2"""
    def __init__(self):
        self.v_template = None
        self.f = None
        self.Jtr = None
        self.J_regressor = None
        self.shapedirs = None
        self.posedirs = None
        
    def charger_modele_star(self, gender='neutral'):
        """COPIE EXACTE de la fonction du script 2"""
        npz_path = os.path.join(STAR_DIR, gender, f"{gender}.npz")
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Modèle STAR non trouvé: {npz_path}")
            
        data = np.load(npz_path)
        self.v_template = data['v_template']
        self.f = data['f']
        self.J_regressor = data['J_regressor']
        self.shapedirs = data.get('shapedirs', None)
        self.posedirs = data.get('posedirs', None)
        self.Jtr = self.J_regressor.dot(self.v_template)
        
        return True
    
    def calculer_mesures_modele(self, vertices, joints, mapping):
        """
        Mesure le vrai périmètre de coupe (comme un mètre ruban)
        """
        mesures = {}

        for mesure, info in mapping.items():
            joint_indices = info["joints"]

            if len(joint_indices) == 2:
                mesures[mesure] = euclidean(joints[joint_indices[0]], joints[joint_indices[1]])

            elif len(joint_indices) == 1:
                joint_pos = joints[joint_indices[0]]
                y_cible = joint_pos[1]

                tolerance = 0.015
                masque = np.abs(vertices[:, 1] - y_cible) < tolerance
                points_coupe = vertices[masque]

                if len(points_coupe) < 20:
                    tolerance = 0.03
                    masque = np.abs(vertices[:, 1] - y_cible) < tolerance
                    points_coupe = vertices[masque]

                if len(points_coupe) < 10:
                    mesures[mesure] = 50.0
                    continue

                points_2d = points_coupe[:, [0, 2]]
                centre = np.mean(points_2d, axis=0)

                angles = np.arctan2(
                    points_2d[:, 1] - centre[1],
                    points_2d[:, 0] - centre[0]
                )
                ordre = np.argsort(angles)
                contour = points_2d[ordre]

                perimeter = 0.0
                n = len(contour)
                for i in range(n):
                    p1 = contour[i]
                    p2 = contour[(i + 1) % n]
                    perimeter += np.linalg.norm(p2 - p1)

                mesures[mesure] = perimeter

        return mesures
    
    def deformer_modele(self, mesures_cibles, mesures_actuelles):
        if self.shapedirs is None:
            print("Pas de blend shapes disponibles")
            return self.v_template, np.zeros(10)

        n_betas = min(10, self.shapedirs.shape[2])

        def objective(betas):
            vertices_deformed = self.v_template + np.sum(
                self.shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2
            )
            joints_deformed = self.J_regressor.dot(vertices_deformed)

            mesures_reelles = self.calculer_mesures_modele(
                vertices_deformed, joints_deformed, DEFAULT_MAPPING
            )

            error = 0.0
            for mesure, valeur_cible in mesures_cibles.items():
                if mesure not in mesures_reelles:
                    continue
                valeur_modele = mesures_reelles[mesure]
                valeur_cible_m = valeur_cible / 100.0
                poids = 2.0 if mesure in ['tour_taille', 'tour_hanches'] else 1.0
                error += poids * ((valeur_modele - valeur_cible_m) ** 2)

            regularization = 0.05 * np.sum(betas ** 2)
            return error + regularization

        initial_betas = np.zeros(n_betas)
        bounds = [(-5, 5)] * n_betas

        result = minimize(
            objective,
            initial_betas,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-9,
                'gtol': 1e-7
            }
        )

        betas = result.x
        vertices_final = self.v_template + np.sum(
            self.shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2
        )

        joints_final = self.J_regressor.dot(vertices_final)
        mesures_finales = self.calculer_mesures_modele(vertices_final, joints_final, DEFAULT_MAPPING)
        print(f"✅ Optimisation terminée: {result.nit} itérations")
        for m, v in mesures_finales.items():
            if m in mesures_cibles:
                print(f"   {m}: obtenu={v*100:.1f}cm | cible={mesures_cibles[m]}cm")

        return vertices_final, betas


class VetementGenerator:
    """CLASSE CORRIGÉE - Jupes ovale et droite corrigées"""
    
    @staticmethod
    def detecter_points_anatomiques(verts):
        """Fonction inchangée"""
        y_vals = verts[:, 1]
        x_vals = verts[:, 0]
        z_vals = verts[:, 2]
        
        y_max = np.max(y_vals)
        y_min = np.min(y_vals)
        hauteur_totale = y_max - y_min
        
        y_tete = y_max - 0.1 * hauteur_totale
        y_epaules = y_max - 0.2 * hauteur_totale
        y_taille = y_max - 0.3 * hauteur_totale
        y_hanches = y_max - 0.45 * hauteur_totale
        y_genoux = y_max - 0.75 * hauteur_totale
        
        def calculer_rayon_a_hauteur(y_target, tolerance=0.05):
            mask = np.abs(y_vals - y_target) < tolerance
            if not np.any(mask):
                return 0.1
            points_niveau = verts[mask]
            distances = np.sqrt(points_niveau[:, 0]**2 + points_niveau[:, 2]**2)
            return np.percentile(distances, 75)
        
        rayon_tete = calculer_rayon_a_hauteur(y_tete)
        rayon_epaules = calculer_rayon_a_hauteur(y_epaules)
        rayon_taille = calculer_rayon_a_hauteur(y_taille)
        rayon_hanches = calculer_rayon_a_hauteur(y_hanches)
        
        distances_radiales = np.sqrt(verts[:, 0]**2 + verts[:, 2]**2)
        seuil_bras = np.percentile(distances_radiales, 85)
        
        return {
            'y_tete': y_tete,
            'y_epaules': y_epaules,
            'y_taille': y_taille,
            'y_hanches': y_hanches,
            'y_genoux': y_genoux,
            'y_min': y_min,
            'y_max': y_max,
            'hauteur_totale': hauteur_totale,
            'rayon_tete': rayon_tete,
            'rayon_epaules': rayon_epaules,
            'rayon_taille': rayon_taille,
            'rayon_hanches': rayon_hanches,
            'seuil_bras': seuil_bras
        }

    @staticmethod
    def creer_mesh_jupe_separe(verts_corps, masque_jupe, couleur_nom):
        """
        ✅ VERSION FINALE - ZÉRO TROU GARANTI (cuisses incluses)
        """
        default_result = {
            'mesh_object': None,
            'points_count': 0,
            'faces_count': 0,
            'couleur_rgb': [128, 128, 128],
            'couleur_normalized': [0.5, 0.5, 0.5]
        }
        
        if not VEDO_AVAILABLE:
            print("Vedo non disponible pour créer le mesh")
            return default_result
        
        try:
            from vedo import Mesh
            
            if not isinstance(verts_corps, np.ndarray):
                print("verts_corps n'est pas un numpy array")
                return default_result
                
            if not isinstance(masque_jupe, np.ndarray):
                print("masque_jupe n'est pas un numpy array")
                return default_result
            
            points_jupe = verts_corps[masque_jupe]
            
            if len(points_jupe) < 100:
                print(f"Pas assez de points jupe: {len(points_jupe)}")
                return default_result
            
            y_vals = points_jupe[:, 1]
            y_min = np.min(y_vals)
            y_max = np.max(y_vals)
            hauteur_jupe = y_max - y_min
            
            n_couches = max(200, int(hauteur_jupe * 400))
            n_points_par_cercle = 128
            
            print(f"🔧 DENSITÉ MAXIMALE: {n_couches} couches × {n_points_par_cercle} points/cercle")
            
            rayons_par_hauteur = []
            for layer_idx in range(n_couches):
                y_layer = y_min + (layer_idx / (n_couches - 1)) * hauteur_jupe if n_couches > 1 else y_min
                
                tolerance = hauteur_jupe / (n_couches * 0.5)
                mask_layer = np.abs(y_vals - y_layer) < tolerance
                
                if np.sum(mask_layer) >= 3:
                    points_layer = points_jupe[mask_layer]
                    rayon_moyen = np.mean(np.sqrt(points_layer[:, 0]**2 + points_layer[:, 2]**2))
                    rayon_moyen *= 1.08
                else:
                    if len(rayons_par_hauteur) > 0:
                        rayon_moyen = rayons_par_hauteur[-1]
                    else:
                        rayon_moyen = np.percentile(np.sqrt(points_jupe[:, 0]**2 + points_jupe[:, 2]**2), 50)
                        rayon_moyen *= 1.08
                
                rayons_par_hauteur.append(rayon_moyen)
            
            couches = []
            
            for layer_idx in range(n_couches):
                y_layer = y_min + (layer_idx / (n_couches - 1)) * hauteur_jupe if n_couches > 1 else y_min
                rayon_cible = rayons_par_hauteur[layer_idx]
                
                angles = np.linspace(0, 2*np.pi, n_points_par_cercle, endpoint=False)
                indices_cercle = []
                
                for angle in angles:
                    x_syn = rayon_cible * np.cos(angle)
                    z_syn = rayon_cible * np.sin(angle)
                    
                    idx_nouveau = len(points_jupe)
                    points_jupe = np.vstack([points_jupe, [[x_syn, y_layer, z_syn]]])
                    indices_cercle.append(idx_nouveau)
                
                couches.append(np.array(indices_cercle))
                
                if layer_idx % 40 == 0:
                    print(f"  ✅ Couche {layer_idx}/{n_couches}: {len(indices_cercle)} pts, y={y_layer:.3f}, r={rayon_cible:.3f}")
            
            if len(couches) < 2:
                print("❌ Pas assez de couches")
                return default_result
            
            faces = []
            
            for layer_idx in range(len(couches) - 1):
                couche_actuelle = couches[layer_idx]
                couche_suivante = couches[layer_idx + 1]
                
                n_curr = len(couche_actuelle)
                n_next = len(couche_suivante)
                
                for i in range(n_curr):
                    i_next = (i + 1) % n_curr
                    j = i % n_next
                    j_next = (i + 1) % n_next
                    
                    faces.append([couche_actuelle[i], couche_suivante[j], couche_actuelle[i_next]])
                    faces.append([couche_actuelle[i_next], couche_suivante[j], couche_suivante[j_next]])
            
            couche_bas = couches[-1]
            centre_bas = np.mean(points_jupe[couche_bas], axis=0)
            centre_bas_idx = len(points_jupe)
            points_jupe = np.vstack([points_jupe, centre_bas])
            
            for i in range(len(couche_bas)):
                i_next = (i + 1) % len(couche_bas)
                faces.append([couche_bas[i], couche_bas[i_next], centre_bas_idx])
            
            couche_haut = couches[0]
            centre_haut = np.mean(points_jupe[couche_haut], axis=0)
            centre_haut_idx = len(points_jupe)
            points_jupe = np.vstack([points_jupe, centre_haut])
            
            for i in range(len(couche_haut)):
                i_next = (i + 1) % len(couche_haut)
                faces.append([centre_haut_idx, couche_haut[i_next], couche_haut[i]])
            
            couleur_rgb = COULEURS_DISPONIBLES.get(couleur_nom, [128, 128, 128])
            couleur_normalized = [c/255.0 for c in couleur_rgb]
            
            if len(faces) > 0:
                try:
                    mesh_jupe = Mesh([points_jupe, faces])
                    mesh_jupe.color(couleur_normalized).alpha(1.0)
                    mesh_jupe.smooth(niter=5)
                    
                    print(f"✅ MESH PARFAIT ZÉRO TROU: {len(points_jupe)} pts, {len(faces)} faces")
                    
                    return {
                        'mesh_object': mesh_jupe,
                        'points_count': len(points_jupe),
                        'faces_count': len(faces),
                        'couleur_rgb': couleur_rgb,
                        'couleur_normalized': couleur_normalized
                    }
                except Exception as e:
                    print(f"❌ Erreur création mesh: {e}")
                    import traceback
                    traceback.print_exc()
                    return default_result
            else:
                print("❌ Aucune face générée")
                return default_result
                
        except Exception as e:
            print(f"❌ Erreur complète mesh jupe: {e}")
            import traceback
            traceback.print_exc()
            return default_result

    @staticmethod
    def lisser_jupe(verts, masque_jupe, iterations=2):
        """
        ✅ Lisse la surface de la jupe
        """
        verts_lisses = verts.copy()
        
        for iteration in range(iterations):
            nouveaux_verts = verts_lisses.copy()
            
            for i, est_jupe in enumerate(masque_jupe):
                if est_jupe:
                    distances = np.linalg.norm(verts_lisses - verts_lisses[i], axis=1)
                    voisins = np.where((distances < 0.03) & (distances > 0))[0]
                    
                    if len(voisins) > 2:
                        poids_centre = 0.75
                        poids_voisins = (1 - poids_centre) / len(voisins)
                        
                        nouveaux_verts[i] = (poids_centre * verts_lisses[i] +
                                        poids_voisins * np.sum(verts_lisses[voisins], axis=0))
            
            verts_lisses = nouveaux_verts
            print(f"✅ Lissage iteration {iteration + 1}/{iterations} terminée")
        
        return verts_lisses

    @staticmethod
    def calculer_profil_jupe_droite(points_anat, longueur_relative):
        """
        ✅ JUPE DROITE CORRIGÉE - Évasement raisonnable, pas excessif
        """
        y_taille = points_anat['y_taille']
        y_hanches = points_anat['y_hanches']
        y_min = points_anat['y_min']
        hauteur_totale = points_anat['hauteur_totale']
        
        rayon_taille = points_anat['rayon_taille']
        rayon_hanches = points_anat['rayon_hanches']
        
        y_debut_jupe = y_taille - 0.02
        y_bas_jupe = y_hanches - (longueur_relative * hauteur_totale)
        y_bas_jupe = max(y_bas_jupe, y_min + 0.1)
        
        rayon_debut = rayon_taille * 1.08
        rayon_hanches_jupe = rayon_hanches * 1.12

        # ✅ CORRECTION: rayon_bas = même que hanches (jupe DROITE = pas d'évasement)
        # Avant: * 1.25 → trop évasé pour une jupe droite
        rayon_bas = rayon_hanches * 1.12

        y_mi_hanches = (y_debut_jupe + y_hanches) / 2
        rayon_mi_hanches = (rayon_debut + rayon_hanches_jupe) / 2
        
        y_mi_cuisses = (y_hanches + y_bas_jupe) / 2
        rayon_mi_cuisses = (rayon_hanches_jupe + rayon_bas) / 2
        
        print(f"👗 PROFIL JUPE DROITE CORRIGÉ:")
        print(f"   ✨ Taille: {rayon_debut:.3f}m (corps={rayon_taille:.3f}m × 1.08)")
        print(f"   💃 Hanches: {rayon_hanches_jupe:.3f}m (corps={rayon_hanches:.3f}m × 1.12)")
        print(f"   🦵 Bas: {rayon_bas:.3f}m (= hanches × 1.12 → DROIT)")
        print(f"   📏 Longueur: {y_debut_jupe - y_bas_jupe:.3f}m")
        
        return {
            'type': 'droite',
            'y_debut': y_debut_jupe,
            'y_bas': y_bas_jupe,
            'rayon_debut': rayon_debut,
            'rayon_hanches_jupe': rayon_hanches_jupe,
            'rayon_bas': rayon_bas,
            'y_hanches': y_hanches,
            'y_mi_hanches': y_mi_hanches,
            'rayon_mi_hanches': rayon_mi_hanches,
            'y_mi_cuisses': y_mi_cuisses,
            'rayon_mi_cuisses': rayon_mi_cuisses
        }

    @staticmethod
    def calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur=1.6):
        """
        ✅ JUPE OVALE CORRIGÉE - Vraiment évasée et ample jusqu'en bas
        Avant: rayon_bas = rayon_hanches * 0.9 → rentrait vers l'intérieur !
        Après: rayon_bas = rayon_hanches * ampleur → reste ample jusqu'en bas
        """
        y_taille = points_anat['y_taille']
        y_hanches = points_anat['y_hanches']
        y_min = points_anat['y_min']
        hauteur_totale = points_anat['hauteur_totale']
        
        rayon_taille = points_anat['rayon_taille']
        rayon_hanches = points_anat['rayon_hanches']
        
        y_debut_jupe = y_taille - 0.03
        y_bas_jupe = y_hanches - (longueur_relative * hauteur_totale)
        y_bas_jupe = max(y_bas_jupe, y_min + 0.1)
        
        # ✅ CORRECTION PRINCIPALE:
        rayon_debut = rayon_taille * 0.90       # Serré à la taille
        rayon_max = rayon_hanches * ampleur     # Maximum aux hanches
        rayon_bas = rayon_hanches * ampleur     # ✅ Reste ample jusqu'en bas
                                                # AVANT: * 0.9 → rentrait dedans !

        y_max_largeur = y_hanches               # ✅ Max dès les hanches (pas -0.1)
        
        print(f"👗 PROFIL JUPE OVALE CORRIGÉ:")
        print(f"   ✨ Taille: {rayon_debut:.3f}m (corps={rayon_taille:.3f}m × 0.90)")
        print(f"   💃 Hanches (max): {rayon_max:.3f}m (corps={rayon_hanches:.3f}m × {ampleur})")
        print(f"   🌸 Bas: {rayon_bas:.3f}m (= hanches × {ampleur} → RESTE AMPLE)")
        print(f"   📏 Longueur: {y_debut_jupe - y_bas_jupe:.3f}m")
        
        return {
            'type': 'ovale',
            'y_debut': y_debut_jupe,
            'y_bas': y_bas_jupe,
            'y_max_largeur': y_max_largeur,
            'rayon_debut': rayon_debut,
            'rayon_max': rayon_max,
            'rayon_bas': rayon_bas,
            'ampleur': ampleur
        }

    @staticmethod
    def calculer_profil_jupe_trapeze(points_anat, longueur_relative, evasement=1.6):
        """✅ Jupe trapèze inchangée"""
        y_taille = points_anat['y_taille']
        y_hanches = points_anat['y_hanches']
        y_min = points_anat['y_min']
        hauteur_totale = points_anat['hauteur_totale']
        
        rayon_taille = points_anat['rayon_taille']
        rayon_hanches = points_anat['rayon_hanches']
        
        y_debut_jupe = y_taille - 0.03
        y_bas_jupe = y_hanches - (longueur_relative * hauteur_totale)
        y_bas_jupe = max(y_bas_jupe, y_min + 0.1)
        
        rayon_debut = rayon_taille * 0.88
        rayon_bas = rayon_hanches * evasement
        
        return {
            'type': 'trapeze',
            'y_debut': y_debut_jupe,
            'y_bas': y_bas_jupe,
            'rayon_debut': rayon_debut,
            'rayon_bas': rayon_bas,
            'evasement': evasement
        }

    @staticmethod
    def calculer_rayon_pour_hauteur(profil_jupe, y):
        """✅ Calcule le rayon à partir des paramètres sauvegardés"""
        type_jupe = profil_jupe['type']
        
        if y > profil_jupe['y_debut']:
            return 0
        elif y < profil_jupe['y_bas']:
            return 0
            
        if type_jupe == 'droite':
            return VetementGenerator._calculer_rayon_droite(profil_jupe, y)
        elif type_jupe == 'ovale':
            return VetementGenerator._calculer_rayon_ovale(profil_jupe, y)
        elif type_jupe == 'trapeze':
            return VetementGenerator._calculer_rayon_trapeze(profil_jupe, y)
        else:
            return 0

    @staticmethod
    def _calculer_rayon_droite(profil, y):
        """
        ✅ JUPE DROITE - Profil uniforme depuis les hanches jusqu'en bas
        """
        y_debut = profil['y_debut']
        y_bas = profil['y_bas']
        y_hanches = profil['y_hanches']
        y_mi_hanches = profil.get('y_mi_hanches', (y_debut + y_hanches) / 2)
        y_mi_cuisses = profil.get('y_mi_cuisses', (y_hanches + y_bas) / 2)
        
        rayon_debut = profil['rayon_debut']
        rayon_mi_hanches = profil.get('rayon_mi_hanches', (profil['rayon_debut'] + profil['rayon_hanches_jupe']) / 2)
        rayon_hanches_jupe = profil['rayon_hanches_jupe']
        rayon_mi_cuisses = profil.get('rayon_mi_cuisses', profil['rayon_hanches_jupe'])
        rayon_bas = profil['rayon_bas']
        
        # Zone 1: TAILLE → MI-HANCHES
        if y >= y_mi_hanches:
            if y_debut == y_mi_hanches:
                return rayon_debut
            t = (y_debut - y) / (y_debut - y_mi_hanches)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            return rayon_debut + t_smooth * (rayon_mi_hanches - rayon_debut)
        
        # Zone 2: MI-HANCHES → HANCHES
        elif y >= y_hanches:
            if y_mi_hanches == y_hanches:
                return rayon_mi_hanches
            t = (y_mi_hanches - y) / (y_mi_hanches - y_hanches)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            return rayon_mi_hanches + t_smooth * (rayon_hanches_jupe - rayon_mi_hanches)
        
        # Zone 3: HANCHES → BAS (✅ DROIT = rayon constant)
        else:
            # Rayon quasi-constant depuis les hanches jusqu'en bas (jupe droite)
            return rayon_hanches_jupe

    @staticmethod
    def _calculer_rayon_ovale(profil, y):
        """
        ✅ JUPE OVALE CORRIGÉE - S'évase depuis la taille, reste ample jusqu'en bas
        Avant: rentrait vers l'intérieur sous les hanches
        Après: reste à rayon_max depuis les hanches jusqu'en bas
        """
        y_debut = profil['y_debut']
        y_bas = profil['y_bas']
        y_max_largeur = profil['y_max_largeur']
        
        rayon_debut = profil['rayon_debut']
        rayon_max = profil['rayon_max']
        rayon_bas = profil['rayon_bas']
        
        # Zone 1: Taille → Hanches (évasement progressif)
        if y >= y_max_largeur:
            if y_debut == y_max_largeur:
                return rayon_debut
            t = (y_debut - y) / (y_debut - y_max_largeur)
            t_curve = 0.5 * (1 - np.cos(np.pi * t))  # Easing progressif
            return rayon_debut + t_curve * (rayon_max - rayon_debut)
        
        # Zone 2: Hanches → Bas
        # ✅ CORRECTION: reste ample avec légère variation naturelle
        # AVANT: t_curve = 0.5 * (1 + np.cos(...)) → rentrait vers l'intérieur !
        else:
            if y_max_largeur == y_bas:
                return rayon_max
            t = (y_max_largeur - y) / (y_max_largeur - y_bas)
            # Légère ondulation naturelle (±5%) pour aspect réaliste
            variation = 0.05 * np.sin(np.pi * t)
            return rayon_max * (1 - variation)

    @staticmethod
    def _calculer_rayon_trapeze(profil, y):
        """✅ Jupe trapèze inchangée - évasement linéaire de haut en bas"""
        y_debut = profil['y_debut']
        y_bas = profil['y_bas']
        
        rayon_debut = profil['rayon_debut']
        rayon_bas = profil['rayon_bas']
        
        if y_debut == y_bas:
            return rayon_debut
        t = (y_debut - y) / (y_debut - y_bas)
        return rayon_debut + t * (rayon_bas - rayon_debut)

    @staticmethod
    def appliquer_forme_jupe(verts, profil_jupe):
        """
        CORRIGÉE - Crée une vraie jupe qui couvre complètement le mannequin
        """
        verts_modifies = verts.copy()
        y_vals = verts[:, 1]
        
        y_debut = profil_jupe['y_debut']
        y_bas = profil_jupe['y_bas']
        
        masque_jupe = (y_vals <= y_debut) & (y_vals >= y_bas)
        
        for i in range(len(verts)):
            if masque_jupe[i]:
                x, y, z = verts[i]
                distance_actuelle = np.sqrt(x**2 + z**2)
                
                if distance_actuelle > 0.001:
                    nouveau_rayon = VetementGenerator.calculer_rayon_pour_hauteur(profil_jupe, y)
                    
                    if nouveau_rayon > 0:
                        facteur = nouveau_rayon / distance_actuelle
                        verts_modifies[i, 0] = x * facteur
                        verts_modifies[i, 2] = z * facteur
        
        y_couches = np.linspace(y_bas, y_debut, max(20, int((y_debut - y_bas) * 100)))
        vertices_synthétiques = []
        
        for y_target in y_couches:
            tolerance = (y_debut - y_bas) / 50
            idx_couche = np.where(np.abs(y_vals - y_target) < tolerance)[0]
            
            if len(idx_couche) < 8:
                rayon_cible = VetementGenerator.calculer_rayon_pour_hauteur(profil_jupe, y_target)
                
                if rayon_cible > 0:
                    n_points = 32
                    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
                    
                    for angle in angles:
                        x_syn = rayon_cible * np.cos(angle)
                        z_syn = rayon_cible * np.sin(angle)
                        vertices_synthétiques.append([x_syn, y_target, z_syn])
        
        if len(vertices_synthétiques) > 0:
            vertices_synthétiques = np.array(vertices_synthétiques)
            verts_modifies = np.vstack([verts_modifies, vertices_synthétiques])
        
        return verts_modifies, masque_jupe


# ===== CLASSE VISUALISATEUR 3D =====
class Visualisateur3D:
    def __init__(self):
        self.active_plotters = []
    
    def calculer_hauteur_mannequin(self, vertices):
        if vertices is None or len(vertices) == 0:
            return 1.7
        
        y_min = np.min(vertices[:, 1])
        y_max = np.max(vertices[:, 1])
        hauteur = y_max - y_min
        
        print(f"📏 Hauteur mannequin calculée: {hauteur:.2f}")
        print(f"   Y min (pieds): {y_min:.2f}")
        print(f"   Y max (tête): {y_max:.2f}")
        
        return hauteur, y_min, y_max
    
    def configurer_camera_pieds_visibles(self, plotter, vertices):
        if vertices is None or len(vertices) == 0:
            camera_position = (0, 0, 4.0)
            look_at_point = (0, 0.5, 0)
            up_vector = (0, 1, 0)
        else:
            hauteur, y_min, y_max = self.calculer_hauteur_mannequin(vertices)
            centre_y_reel = (y_min + y_max) / 2.0
            distance_camera = max(3.5, hauteur * 2.5)
            camera_position = (0, centre_y_reel * 0.8, distance_camera)
            look_at_point = (0, centre_y_reel, 0)
            up_vector = (0, 1, 0)
        
        plotter.camera.SetPosition(*camera_position)
        plotter.camera.SetFocalPoint(*look_at_point)
        plotter.camera.SetViewUp(*up_vector)
        plotter.camera.SetViewAngle(55)
        plotter.camera.SetClippingRange(0.1, 15.0)
        plotter.reset_camera()
        plotter.camera.Zoom(0.8)
        
        return True
        
    def capturer_mannequin_3d_vedo(self, vertices, faces, fichier_sortie, titre="Mannequin"):
        if not VEDO_AVAILABLE:
            print("❌ Vedo non disponible")
            return False
        
        try:
            mesh_mannequin = Mesh([vertices, faces])
            mesh_mannequin.color([0.8, 0.6, 0.4]).alpha(0.9)
            
            plt = Plotter(bg='white', axes=0, interactive=False, offscreen=True, size=(1920, 1920))
            plt.add(mesh_mannequin)
            self.configurer_camera_pieds_visibles(plt, vertices)
            plt.render()
            img_array = plt.screenshot(filename=fichier_sortie, scale=3)
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur capture mannequin: {e}")
            return False
    
    def capturer_rendu_3d_vedo(self, vertices_corps, faces, masque_vetement,
                                vertices_vetement, mesh_vetement_data,
                                fichier_sortie, titre="Mannequin avec Vêtement"):
        if not VEDO_AVAILABLE:
            print("❌ Vedo non disponible")
            return False
        
        try:
            mesh_corps = Mesh([vertices_corps, faces])
            mesh_corps.color([0.8, 0.6, 0.4]).alpha(0.8)
            
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data is not None and mesh_vetement_data['mesh_object'] is not None:
                meshes_a_afficher.append(mesh_vetement_data['mesh_object'])
            
            plt = Plotter(bg='white', axes=0, interactive=False, offscreen=True, size=(1920, 1920))
            
            for mesh in meshes_a_afficher:
                plt.add(mesh)
            
            self.configurer_camera_pieds_visibles(plt, vertices_corps)
            plt.render()
            img_array = plt.screenshot(filename=fichier_sortie, scale=3)
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur capture vêtement: {e}")
            return False
    
    def afficher_mannequin_avec_vetement(self, vertices_corps, faces, masque_vetement,
                                        vertices_vetement, mesh_vetement_data, titre="Mannequin avec Vêtement"):
        if not VEDO_AVAILABLE:
            return False
        
        try:
            mesh_corps = Mesh([vertices_corps, faces])
            mesh_corps.color([0.8, 0.6, 0.4]).alpha(0.8)
            
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data is not None and mesh_vetement_data['mesh_object'] is not None:
                meshes_a_afficher.append(mesh_vetement_data['mesh_object'])
            
            def lancer_vedo():
                try:
                    plt = Plotter(bg='white', axes=1, title=titre, interactive=True)
                    
                    for mesh in meshes_a_afficher:
                        plt.add(mesh)
                    
                    self.configurer_camera_pieds_visibles(plt, vertices_corps)
                    plt.show(interactive=True)
                    plt.close()
                    
                except Exception as e:
                    print(f"❌ Erreur Vedo thread: {e}")
            
            thread = threading.Thread(target=lancer_vedo, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur affichage: {e}")
            return False


def cleanup_temp_files():
    for directory in [TEMP_DIR, PREVIEW_DIR]:
        temp_files = glob.glob(os.path.join(directory, "*.*"))
        for file in temp_files:
            try:
                if os.path.getmtime(file) < time.time() - 3600:
                    os.remove(file)
            except:
                pass

atexit.register(cleanup_temp_files)

mannequin_gen = MannequinGenerator()
visualisateur = Visualisateur3D()

TIMEOUT_GENERATION = 180
TIMEOUT_CAPTURE = 60
TIMEOUT_SCENARIOS = 300

def timeout_function(timeout_duration):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"❌ Erreur dans {func.__name__}: {e}")
                raise e
        return wrapper
    return decorator

def simple_timeout_function(timeout_duration):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > timeout_duration:
                    print(f"⚠️ {func.__name__} a pris {elapsed:.1f}s (>{timeout_duration}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ {func.__name__} a échoué après {elapsed:.1f}s: {e}")
                raise e
        return wrapper
    return decorator


def valider_type_vetement(type_saisi):
    if not type_saisi or not isinstance(type_saisi, str):
        types_disponibles = list(TYPES_VETEMENTS.keys())
        return None, types_disponibles
    
    type_saisi_lower = type_saisi.lower().strip()
    
    for type_officiel in TYPES_VETEMENTS.keys():
        if type_officiel.lower() == type_saisi_lower:
            return type_officiel
    
    types_disponibles = list(TYPES_VETEMENTS.keys())
    return None, types_disponibles


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_environment():
    return os.environ.get('RENDER', 'false') == 'true' or \
           os.environ.get('ENVIRONMENT', 'development') == 'production'

def get_base_url():
    if get_environment():
        return "https://star-api-docker.onrender.com"
    else:
        local_ip = get_local_ip()
        return f"http://{local_ip}:5000"


IS_PRODUCTION = get_environment()
BASE_URL = get_base_url()

print(f"🌍 Environnement: {'PRODUCTION' if IS_PRODUCTION else 'DÉVELOPPEMENT'}")
print(f"🔗 Base URL: {BASE_URL}")


@app.route('/api/mannequin/webview/<mannequin_id>', methods=['GET'])
def generate_mannequin_webview(mannequin_id):
    try:
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        gender = str(data['gender'])
        
        html_content = generer_html_mannequin_3d(vertices, faces, gender)
        
        html_file = os.path.join(PREVIEW_DIR, f"{mannequin_id}_webview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        webview_url = f"{BASE_URL}/webview/{mannequin_id}_webview.html"
        
        return jsonify({
            'success': True,
            'url': webview_url,
            'visualization_url': webview_url,
            'environment': 'production' if IS_PRODUCTION else 'development',
            'base_url': BASE_URL,
            'debug_info': {
                'vertices_count': len(vertices),
                'faces_count': len(faces),
                'gender': gender,
                'file_path': html_file
            },
            'message': f'Visualisation WebView générée pour mannequin {gender}'
        })
        
    except Exception as e:
        print(f"❌ Erreur génération WebView mannequin: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vetement/webview/<vetement_id>', methods=['GET'])
def generate_vetement_webview(vetement_id):
    try:
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Vêtement non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices_corps = data['vertices_corps']
        vertices_avec_vetement = data['vertices_avec_vetement']
        faces = data['faces']
        masque_vetement = data['masque_vetement']
        couleur = str(data['couleur'])
        type_vetement = str(data['type_vetement'])
        
        html_content = generer_html_vetement_3d(
            vertices_corps, vertices_avec_vetement, faces,
            masque_vetement, couleur, type_vetement
        )
        
        html_file = os.path.join(PREVIEW_DIR, f"{vetement_id}_webview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        webview_url = f"{BASE_URL}/webview/{vetement_id}_webview.html"
        
        return jsonify({
            'success': True,
            'url': webview_url,
            'visualization_url': webview_url,
            'environment': 'production' if IS_PRODUCTION else 'development',
            'base_url': BASE_URL,
            'message': f'Visualisation WebView générée pour {type_vetement} {couleur}'
        })
        
    except Exception as e:
        print(f"❌ Erreur génération WebView vêtement: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/webview/test', methods=['GET'])
def test_webview():
    test_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test WebView - {'Production' if IS_PRODUCTION else 'Développement'}</title>
    <style>
        body {{
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: Arial, sans-serif; color: white;
            text-align: center; min-height: 100vh;
            display: flex; flex-direction: column;
            justify-content: center; align-items: center;
        }}
        .test-box {{
            background: rgba(255,255,255,0.2); padding: 20px;
            border-radius: 10px; backdrop-filter: blur(10px);
        }}
    </style>
</head>
<body>
    <div class="test-box">
        <h1>✅ WebView Test</h1>
        <p>Environnement: <strong>{'PRODUCTION' if IS_PRODUCTION else 'DÉVELOPPEMENT'}</strong></p>
        <p>URL: <strong>{BASE_URL}</strong></p>
        <div id="timer">Loading...</div>
    </div>
    <script>
        let seconds = 0;
        function updateTimer() {{
            document.getElementById('timer').textContent = `Temps écoulé: ${{seconds}} secondes`;
            seconds++;
        }}
        setInterval(updateTimer, 1000);
        updateTimer();
    </script>
</body>
</html>
    """
    
    test_file = os.path.join(PREVIEW_DIR, "test_webview.html")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_html)
    
    test_url = f"{BASE_URL}/webview/test_webview.html"
    
    return jsonify({
        'success': True,
        'url': test_url,
        'test_url': test_url,
        'environment': 'production' if IS_PRODUCTION else 'development',
        'base_url': BASE_URL,
        'message': f'Page de test WebView générée'
    })


@app.route('/api/debug/urls', methods=['GET'])
def debug_urls():
    return jsonify({
        'environment': 'production' if IS_PRODUCTION else 'development',
        'base_url': BASE_URL,
        'webview_base': f"{BASE_URL}/webview/",
        'test_urls': {
            'test_webview': f"{BASE_URL}/webview/test_webview.html",
            'mannequin_example': f"{BASE_URL}/webview/MANNEQUIN_ID_webview.html",
            'vetement_example': f"{BASE_URL}/webview/VETEMENT_ID_webview.html"
        },
        'render_env': os.environ.get('RENDER', 'Not set'),
        'custom_env': os.environ.get('ENVIRONMENT', 'Not set')
    })


@app.route('/', methods=['GET'])
def test():
    return jsonify({'test': 'helle'})


@app.route('/api/data/<model_id>', methods=['GET'])
def get_model_data(model_id):
    try:
        temp_file = os.path.join(TEMP_DIR, f"{model_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Modèle non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        
        result = {
            'model_id': model_id,
            'vertices_count': len(data['vertices']) if 'vertices' in data else 0,
            'faces_count': len(data['faces']) if 'faces' in data else 0,
            'has_vertices': 'vertices' in data,
            'has_faces': 'faces' in data,
            'data_keys': list(data.keys())
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/webview/<filename>')
def serve_webview_file(filename):
    return send_file(
        os.path.join(PREVIEW_DIR, filename),
        mimetype='text/html'
    )


def convertir_numpy_pour_json(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_numpy_pour_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_numpy_pour_json(item) for item in obj]
    else:
        return obj


def generer_html_mannequin_3d(vertices, faces, gender):
    vertices_json = json.dumps(vertices.tolist())
    faces_json = json.dumps(faces.tolist())
    
    html_template = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mannequin {gender} - Debug</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: Arial, sans-serif; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #debug {{ position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px; z-index: 100; font-size: 12px; max-width: 300px; }}
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 18px; z-index: 50; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading">Chargement du mannequin...</div>
        <div id="debug">
            <div><strong>Debug Info:</strong></div>
            <div id="vertices-info">Vertices: {len(vertices)}</div>
            <div id="faces-info">Faces: {len(faces)}</div>
            <div id="gender-info">Genre: {gender}</div>
            <div id="status">Status: Initialisation...</div>
        </div>
    </div>
    <script>
        function updateStatus(message) {{
            document.getElementById('status').textContent = 'Status: ' + message;
        }}
        try {{
            updateStatus('Chargement des données...');
            if (typeof THREE === 'undefined') throw new Error('Three.js non chargé');
            updateStatus('Three.js OK');
            const vertices = {vertices_json};
            const faces = {faces_json};
            updateStatus(`Données chargées: ${{vertices.length}} vertices`);
            let scene, camera, renderer, mesh;
            function init() {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f0f0);
                camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 15);
                const yValues = vertices.map(v => v[1]);
                const yMin = Math.min(...yValues);
                const yMax = Math.max(...yValues);
                const hauteur = yMax - yMin;
                const centreY = (yMin + yMax) / 2;
                const distance = Math.max(3.5, hauteur * 2.5);
                camera.position.set(0, centreY * 0.8, distance);
                camera.lookAt(0, centreY, 0);
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.shadowMap.enabled = true;
                document.getElementById('container').appendChild(renderer.domElement);
                createMesh();
                setupLighting();
                setupControls();
                document.getElementById('loading').style.display = 'none';
                animate();
                window.addEventListener('resize', onWindowResize);
            }}
            function createMesh() {{
                const geometry = new THREE.BufferGeometry();
                const verticesArray = new Float32Array(vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
                const facesArray = new Uint32Array(faces.flat());
                geometry.setIndex(new THREE.BufferAttribute(facesArray, 1));
                geometry.computeVertexNormals();
                const material = new THREE.MeshLambertMaterial({{ color: 0xcc9966 }});
                mesh = new THREE.Mesh(geometry, material);
                mesh.castShadow = true;
                scene.add(mesh);
            }}
            function setupLighting() {{
                scene.add(new THREE.AmbientLight(0x404040, 0.6));
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 10, 5);
                scene.add(directionalLight);
            }}
            function setupControls() {{
                let isRotating = false;
                let previousTouch = null;
                renderer.domElement.addEventListener('touchstart', (e) => {{ e.preventDefault(); isRotating = true; if (e.touches.length === 1) previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }});
                renderer.domElement.addEventListener('touchmove', (e) => {{ e.preventDefault(); if (isRotating && previousTouch && e.touches.length === 1) {{ const deltaX = e.touches[0].clientX - previousTouch.x; const deltaY = e.touches[0].clientY - previousTouch.y; if (mesh) {{ mesh.rotation.y += deltaX * 0.01; mesh.rotation.x += deltaY * 0.01; }} previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }} }});
                renderer.domElement.addEventListener('touchend', (e) => {{ e.preventDefault(); isRotating = false; previousTouch = null; }});
                let isMouseDown = false; let previousMouse = null;
                renderer.domElement.addEventListener('mousedown', (e) => {{ isMouseDown = true; previousMouse = {{ x: e.clientX, y: e.clientY }}; }});
                renderer.domElement.addEventListener('mousemove', (e) => {{ if (isMouseDown && previousMouse) {{ const deltaX = e.clientX - previousMouse.x; const deltaY = e.clientY - previousMouse.y; if (mesh) {{ mesh.rotation.y += deltaX * 0.01; mesh.rotation.x += deltaY * 0.01; }} previousMouse = {{ x: e.clientX, y: e.clientY }}; }} }});
                renderer.domElement.addEventListener('mouseup', () => {{ isMouseDown = false; previousMouse = null; }});
            }}
            function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
            function onWindowResize() {{ camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }}
            init();
        }} catch (error) {{
            updateStatus('ERREUR: ' + error.message);
            document.getElementById('loading').innerHTML = '<div style="color: red;">Erreur: ' + error.message + '</div>';
        }}
    </script>
</body>
</html>
    """
    return html_template


def generer_html_vetement_3d(vertices_corps, vertices_avec_vetement, faces, masque_vetement, couleur, type_vetement):
    import json
    
    vertices_corps_json = json.dumps(vertices_corps.tolist())
    vertices_vetement_json = json.dumps(vertices_avec_vetement.tolist())
    faces_json = json.dumps(faces.tolist())
    masque_json = json.dumps(masque_vetement.tolist())
    
    couleur_rgb = COULEURS_DISPONIBLES.get(couleur, [128, 128, 128])
    couleur_normalized = [c/255.0 for c in couleur_rgb]
    couleur_normalized_json = json.dumps(couleur_normalized)
    
    html_template = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{type_vetement} {couleur} - Visualisation 3D</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: Arial, sans-serif; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 10px; z-index: 100; font-size: 14px; max-width: 300px; }}
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 18px; z-index: 50; text-align: center; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading"><div>Chargement du vêtement...</div><div style="font-size: 14px; margin-top: 10px;">{type_vetement} {couleur}</div></div>
        <div id="info"><div><strong>🎭 {type_vetement}</strong></div><div style="display:flex;align-items:center;gap:8px;margin-top:4px;"><div style="width:20px;height:20px;border-radius:50%;background:rgb({couleur_rgb[0]},{couleur_rgb[1]},{couleur_rgb[2]});border:2px solid white;"></div>{couleur}</div></div>
    </div>
    <script>
        const verticesCorps = {vertices_corps_json};
        const verticesVetement = {vertices_vetement_json};
        const faces = {faces_json};
        const masqueVetement = {masque_json};
        const couleurVetement = {couleur_normalized_json};
        let scene, camera, renderer;
        let meshCorps, meshVetement;
        function init() {{
            try {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f0f0);
                camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 15);
                const yValues = verticesCorps.map(v => v[1]);
                const yMin = Math.min(...yValues);
                const yMax = Math.max(...yValues);
                const hauteur = yMax - yMin;
                const centreY = (yMin + yMax) / 2;
                const distance = Math.max(3.5, hauteur * 2.5);
                camera.position.set(0, centreY * 0.8, distance);
                camera.lookAt(0, centreY, 0);
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.shadowMap.enabled = true;
                document.getElementById('container').appendChild(renderer.domElement);
                createMeshes();
                setupLighting();
                setupControls();
                document.getElementById('loading').style.display = 'none';
                animate();
                window.addEventListener('resize', onWindowResize);
            }} catch (error) {{
                console.error('❌ Erreur initialisation:', error);
                document.getElementById('loading').innerHTML = '<div style="color: red;">Erreur: ' + error.message + '</div>';
            }}
        }}
        function createMeshes() {{
            const geometryCorps = new THREE.BufferGeometry();
            geometryCorps.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verticesCorps.flat()), 3));
            geometryCorps.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.flat()), 1));
            geometryCorps.computeVertexNormals();
            meshCorps = new THREE.Mesh(geometryCorps, new THREE.MeshLambertMaterial({{ color: 0xcc9966, transparent: true, opacity: 0.8 }}));
            meshCorps.castShadow = true;
            scene.add(meshCorps);
            const verticesVetementOnly = [];
            const facesVetementOnly = [];
            const indexMap = new Map();
            let newIndex = 0;
            for (let i = 0; i < masqueVetement.length; i++) {{
                if (masqueVetement[i]) {{ verticesVetementOnly.push(...verticesVetement[i]); indexMap.set(i, newIndex); newIndex++; }}
            }}
            for (let i = 0; i < faces.length; i++) {{
                const face = faces[i];
                if (masqueVetement[face[0]] && masqueVetement[face[1]] && masqueVetement[face[2]]) {{
                    const nv1 = indexMap.get(face[0]); const nv2 = indexMap.get(face[1]); const nv3 = indexMap.get(face[2]);
                    if (nv1 !== undefined && nv2 !== undefined && nv3 !== undefined) facesVetementOnly.push([nv1, nv2, nv3]);
                }}
            }}
            if (verticesVetementOnly.length > 0 && facesVetementOnly.length > 0) {{
                const geometryVetement = new THREE.BufferGeometry();
                geometryVetement.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verticesVetementOnly), 3));
                geometryVetement.setIndex(new THREE.BufferAttribute(new Uint32Array(facesVetementOnly.flat()), 1));
                geometryVetement.computeVertexNormals();
                meshVetement = new THREE.Mesh(geometryVetement, new THREE.MeshLambertMaterial({{ color: new THREE.Color(couleurVetement[0], couleurVetement[1], couleurVetement[2]), transparent: true, opacity: 0.95 }}));
                meshVetement.castShadow = true;
                scene.add(meshVetement);
            }}
        }}
        function setupLighting() {{
            scene.add(new THREE.AmbientLight(0x404040, 0.4));
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 5);
            scene.add(directionalLight);
            const light1 = new THREE.PointLight(0xffffff, 0.3); light1.position.set(-5, 5, 5); scene.add(light1);
            const light2 = new THREE.PointLight(0xffffff, 0.3); light2.position.set(5, 5, -5); scene.add(light2);
        }}
        function setupControls() {{
            let isRotating = false; let previousTouch = null;
            renderer.domElement.addEventListener('touchstart', (e) => {{ e.preventDefault(); isRotating = true; if (e.touches.length === 1) previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }});
            renderer.domElement.addEventListener('touchmove', (e) => {{ e.preventDefault(); if (isRotating && previousTouch && e.touches.length === 1) {{ const deltaX = e.touches[0].clientX - previousTouch.x; const deltaY = e.touches[0].clientY - previousTouch.y; if (meshCorps) {{ meshCorps.rotation.y += deltaX * 0.01; meshCorps.rotation.x += deltaY * 0.01; }} if (meshVetement) {{ meshVetement.rotation.y += deltaX * 0.01; meshVetement.rotation.x += deltaY * 0.01; }} previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }} }});
            renderer.domElement.addEventListener('touchend', (e) => {{ e.preventDefault(); isRotating = false; previousTouch = null; }});
            let isMouseDown = false; let previousMouse = null;
            renderer.domElement.addEventListener('mousedown', (e) => {{ isMouseDown = true; previousMouse = {{ x: e.clientX, y: e.clientY }}; }});
            renderer.domElement.addEventListener('mousemove', (e) => {{ if (isMouseDown && previousMouse) {{ const deltaX = e.clientX - previousMouse.x; const deltaY = e.clientY - previousMouse.y; if (meshCorps) {{ meshCorps.rotation.y += deltaX * 0.01; meshCorps.rotation.x += deltaY * 0.01; }} if (meshVetement) {{ meshVetement.rotation.y += deltaX * 0.01; meshVetement.rotation.x += deltaY * 0.01; }} previousMouse = {{ x: e.clientX, y: e.clientY }}; }} }});
            renderer.domElement.addEventListener('mouseup', () => {{ isMouseDown = false; previousMouse = null; }});
        }}
        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        function onWindowResize() {{ camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }}
        init();
    </script>
</body>
</html>
    """
    return html_template


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'message': '✅ API Mannequin - Jupes corrigées',
        'star_models_available': os.path.exists(STAR_DIR),
        'couleurs_disponibles': len(COULEURS_DISPONIBLES),
        'types_vetements': len(TYPES_VETEMENTS),
        'vedo_available': VEDO_AVAILABLE,
        'corrections_jupes': {
            'jupe_ovale': '✅ rayon_bas = rayon_hanches × ampleur (était × 0.9 → rentrait !)',
            'jupe_droite': '✅ rayon_bas = rayon_hanches × 1.12 (était × 1.25 → trop évasé)',
            'ampleur_ovale_mini': '1.5 (était 1.3)',
            'ampleur_ovale_genou': '1.6 (était 1.4)',
        }
    })


@app.route('/api/vetement/generate', methods=['POST'])
@timeout_function(TIMEOUT_GENERATION)
def generate_vetement():
    """✅ Route COMPLÈTE avec lissage et mesh séparé"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données JSON requises'}), 400
        
        type_vetement_saisi = data.get('type_vetement', '')
        couleur = data.get('couleur', 'Noir')
        gender = data.get('gender', 'neutral')
        mesures = data.get('mesures', {})
        mannequin_id = data.get('mannequin_id', None)
        
        print(f"👗 ✅ Génération vêtement: {type_vetement_saisi} {couleur} pour {gender}")
        
        result = valider_type_vetement(type_vetement_saisi)
        if isinstance(result, tuple):
            type_vetement_valide, types_disponibles = result
            return jsonify({
                'error': f'Type de vêtement non valide: {type_vetement_saisi}',
                'types_disponibles': types_disponibles
            }), 400
        else:
            type_vetement = result
        
        if couleur not in COULEURS_DISPONIBLES:
            return jsonify({
                'error': f'Couleur non valide: {couleur}',
                'couleurs_disponibles': list(COULEURS_DISPONIBLES.keys())
            }), 400
        
        if mannequin_id:
            temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
            if not os.path.exists(temp_file):
                return jsonify({'error': f'Mannequin {mannequin_id} non trouvé'}), 404
            
            data_mannequin = np.load(temp_file, allow_pickle=True)
            vertices_base = data_mannequin['vertices']
            faces = data_mannequin['faces']
            gender = str(data_mannequin['gender'])
        else:
            mesures_default = {
                'tour_taille': 68, 'tour_hanches': 92,
                'tour_poitrine': 88, 'hauteur': 170, 'longueur_bras': 60
            }
            mesures = {**mesures_default, **mesures}
            
            try:
                if not mannequin_gen.charger_modele_star(gender):
                    return jsonify({'error': 'Impossible de charger le modèle STAR'}), 500
            except Exception as e:
                return jsonify({'error': f'Erreur chargement STAR: {str(e)}'}), 500
            
            vertices_base = mannequin_gen.v_template.copy()
            faces = mannequin_gen.f
            
            if mannequin_gen.shapedirs is not None:
                try:
                    joints_base = mannequin_gen.J_regressor.dot(vertices_base)
                    mesures_actuelles = mannequin_gen.calculer_mesures_modele(vertices_base, joints_base, DEFAULT_MAPPING)
                    vertices_base, betas = mannequin_gen.deformer_modele(mesures, mesures_actuelles)
                except Exception as e:
                    print(f"⚠️ Erreur déformation: {e}")
        
        params_vetement = TYPES_VETEMENTS[type_vetement]
        longueur_relative = params_vetement['longueur_relative']
        
        points_anat = VetementGenerator.detecter_points_anatomiques(vertices_base)
        
        if params_vetement['type'] == 'droite':
            profil = VetementGenerator.calculer_profil_jupe_droite(points_anat, longueur_relative)
        elif params_vetement['type'] == 'ovale':
            ampleur = params_vetement.get('ampleur', 1.6)
            profil = VetementGenerator.calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur)
        elif params_vetement['type'] == 'trapeze':
            evasement = params_vetement.get('evasement', 1.6)
            profil = VetementGenerator.calculer_profil_jupe_trapeze(points_anat, longueur_relative, evasement)
        else:
            return jsonify({'error': f'Type non supporté: {params_vetement["type"]}'}), 400
        
        vertices_avec_vetement, masque_vetement = VetementGenerator.appliquer_forme_jupe(vertices_base, profil)
        
        print(f"🔄 Application du lissage...")
        vertices_avec_vetement = VetementGenerator.lisser_jupe(vertices_avec_vetement, masque_vetement, iterations=2)
        
        print(f"🎨 Création du mesh vêtement séparé...")
        try:
            mesh_vetement_data = VetementGenerator.creer_mesh_jupe_separe(
                vertices_avec_vetement, masque_vetement, couleur
            )
            if not isinstance(mesh_vetement_data, dict):
                mesh_vetement_data = {
                    'mesh_object': None, 'points_count': 0, 'faces_count': 0,
                    'couleur_rgb': [128, 128, 128], 'couleur_normalized': [0.5, 0.5, 0.5]
                }
        except Exception as e:
            print(f"❌ Erreur création mesh: {e}")
            mesh_vetement_data = {
                'mesh_object': None, 'points_count': 0, 'faces_count': 0,
                'couleur_rgb': [128, 128, 128], 'couleur_normalized': [0.5, 0.5, 0.5]
            }
        
        import random, string, time as time_module
        timestamp = str(int(time_module.time()))[-6:]
        random_suffix = ''.join(random.choices(string.digits, k=3))
        vetement_id = f"vetement_{type_vetement.lower().replace(' ', '_').replace('-', '_')}_{couleur.lower()}_{timestamp}_{random_suffix}"
        
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        try:
            np.savez_compressed(
                temp_file,
                vertices_corps=vertices_base,
                vertices_avec_vetement=vertices_avec_vetement,
                faces=faces,
                masque_vetement=masque_vetement,
                couleur=couleur,
                type_vetement=type_vetement,
                gender=gender,
                mesures=mesures,
                longueur_relative=longueur_relative,
                profil=profil,
                mesh_points_count=mesh_vetement_data.get('points_count', 0),
                mesh_faces_count=mesh_vetement_data.get('faces_count', 0),
                mesh_couleur_rgb=mesh_vetement_data.get('couleur_rgb', [128, 128, 128]),
                mesh_couleur_normalized=mesh_vetement_data.get('couleur_normalized', [0.5, 0.5, 0.5]),
                mesh_created=mesh_vetement_data.get('mesh_object') is not None
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Erreur sauvegarde: {str(e)}'}), 500
        
        info = {
            'type_vetement': type_vetement,
            'couleur': couleur,
            'gender': gender,
            'vertices_count': len(vertices_avec_vetement),
            'faces_count': len(faces),
            'vetement_vertices': int(np.sum(masque_vetement)),
            'longueur_relative': float(longueur_relative),
            'mesures_appliquees': mesures,
            'lissage_applique': True,
            'mesh_separe_cree': mesh_vetement_data.get('mesh_object') is not None,
            'mesh_points': mesh_vetement_data.get('points_count', 0),
            'mesh_faces': mesh_vetement_data.get('faces_count', 0),
            'couleur_rgb': mesh_vetement_data.get('couleur_rgb', [128, 128, 128]),
        }
        
        info_convertie = convertir_numpy_pour_json(info)
        
        return jsonify({
            'success': True,
            'message': f'✅ {type_vetement} {couleur} généré avec corrections jupes',
            'info': info_convertie,
            'vetement_id': vetement_id,
            'corrections_appliquees': {
                'jupe_ovale': '✅ Reste ample jusqu\'en bas (rayon_bas = rayon_max)',
                'jupe_droite': '✅ Profil uniforme sans évasement excessif',
            }
        })
        
    except Exception as e:
        print(f"❌ Erreur génération vêtement: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generer_html_vetement_preview(vertices_corps, vertices_avec_vetement, faces, masque_vetement, couleur, type_vetement):
    import json
    
    vertices_corps_json = json.dumps(vertices_corps.tolist())
    vertices_vetement_json = json.dumps(vertices_avec_vetement.tolist())
    faces_json = json.dumps(faces.tolist())
    masque_json = json.dumps(masque_vetement.tolist())
    
    couleur_rgb = COULEURS_DISPONIBLES.get(couleur, [128, 128, 128])
    couleur_normalized = [c/255.0 for c in couleur_rgb]
    couleur_json = json.dumps(couleur_normalized)
    
    html_template = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{type_vetement} {couleur}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{ width: 100vw; height: 100vh; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }}
        #info {{ position: absolute; top: 15px; left: 15px; background: rgba(0,0,0,0.85); backdrop-filter: blur(10px); color: white; padding: 16px 20px; border-radius: 12px; z-index: 10; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
        .info-title {{ font-size: 18px; font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; }}
        .color-swatch {{ display: inline-block; width: 24px; height: 24px; border-radius: 6px; background-color: rgb({couleur_rgb[0]},{couleur_rgb[1]},{couleur_rgb[2]}); box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 2px solid rgba(255,255,255,0.3); }}
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.9); color: white; padding: 30px 40px; border-radius: 16px; text-align: center; z-index: 100; }}
        .spinner {{ border: 3px solid rgba(255,255,255,0.1); border-radius: 50%; border-top: 3px solid white; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div id="loading"><div class="spinner"></div><div style="font-size: 16px; font-weight: 600;">Chargement...</div><div style="font-size: 13px; margin-top: 8px; opacity: 0.7;">{type_vetement} {couleur}</div></div>
    <div id="info"><div class="info-title"><span class="color-swatch"></span><span>{type_vetement}</span></div></div>
    <script>
        let scene, camera, renderer;
        let meshCorps, meshVetement;
        let rotationX = 0, rotationY = 0;
        const verticesCorps = {vertices_corps_json};
        const verticesVetement = {vertices_vetement_json};
        const faces = {faces_json};
        const masque = {masque_json};
        const couleur = {couleur_json};
        function init() {{
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf5f7fa);
            const yValues = verticesCorps.map(v => v[1]);
            const yMin = Math.min(...yValues); const yMax = Math.max(...yValues);
            const hauteur = yMax - yMin; const centreY = (yMin + yMax) / 2;
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
            const dist = Math.max(2.2, hauteur * 2.4);
            camera.position.set(0.3, centreY * 0.85, dist);
            camera.lookAt(0, centreY, 0);
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.2;
            document.body.appendChild(renderer.domElement);
            const geometryCorps = new THREE.BufferGeometry();
            geometryCorps.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verticesCorps.flat()), 3));
            geometryCorps.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.flat()), 1));
            geometryCorps.computeVertexNormals();
            meshCorps = new THREE.Mesh(geometryCorps, new THREE.MeshStandardMaterial({{ color: 0xd4a574, roughness: 0.65, metalness: 0.05 }}));
            meshCorps.castShadow = true; meshCorps.receiveShadow = true;
            scene.add(meshCorps);
            const verticesVetOnly = []; const facesVetOnly = [];
            const indexMap = new Map(); let newIdx = 0;
            for (let i = 0; i < masque.length; i++) {{ if (masque[i]) {{ verticesVetOnly.push(...verticesVetement[i]); indexMap.set(i, newIdx); newIdx++; }} }}
            for (let i = 0; i < faces.length; i++) {{ const [v1, v2, v3] = faces[i]; if (masque[v1] && masque[v2] && masque[v3]) {{ const nv1 = indexMap.get(v1); const nv2 = indexMap.get(v2); const nv3 = indexMap.get(v3); if (nv1 !== undefined && nv2 !== undefined && nv3 !== undefined) facesVetOnly.push([nv1, nv2, nv3]); }} }}
            if (verticesVetOnly.length > 0 && facesVetOnly.length > 0) {{
                const geometryVet = new THREE.BufferGeometry();
                geometryVet.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verticesVetOnly), 3));
                geometryVet.setIndex(new THREE.BufferAttribute(new Uint32Array(facesVetOnly.flat()), 1));
                geometryVet.computeVertexNormals();
                meshVetement = new THREE.Mesh(geometryVet, new THREE.MeshStandardMaterial({{ color: new THREE.Color(couleur[0], couleur[1], couleur[2]), roughness: 0.45, metalness: 0.1, side: THREE.DoubleSide }}));
                meshVetement.castShadow = true; meshVetement.receiveShadow = true;
                scene.add(meshVetement);
            }}
            scene.add(new THREE.AmbientLight(0xffffff, 0.4));
            const mainLight = new THREE.DirectionalLight(0xffffff, 1.2); mainLight.position.set(3, 8, 5); mainLight.castShadow = true; scene.add(mainLight);
            const fillLight = new THREE.DirectionalLight(0xffd9b3, 0.6); fillLight.position.set(-4, 3, 3); scene.add(fillLight);
            const rimLight = new THREE.DirectionalLight(0xb3d9ff, 0.5); rimLight.position.set(0, 2, -5); scene.add(rimLight);
            setupControls();
            document.getElementById('loading').style.display = 'none';
            animate();
            window.addEventListener('resize', onWindowResize);
        }}
        function setupControls() {{
            let isRotating = false; let previousTouch = null;
            renderer.domElement.addEventListener('touchstart', (e) => {{ e.preventDefault(); isRotating = true; if (e.touches.length === 1) previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }});
            renderer.domElement.addEventListener('touchmove', (e) => {{ e.preventDefault(); if (isRotating && previousTouch && e.touches.length === 1) {{ const deltaX = e.touches[0].clientX - previousTouch.x; const deltaY = e.touches[0].clientY - previousTouch.y; rotationY += deltaX * 0.008; rotationX += deltaY * 0.008; rotationX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotationX)); updateRotation(); previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }} }});
            renderer.domElement.addEventListener('touchend', (e) => {{ e.preventDefault(); isRotating = false; previousTouch = null; }});
            let isMouseDown = false; let previousMouse = null;
            renderer.domElement.addEventListener('mousedown', (e) => {{ isMouseDown = true; previousMouse = {{ x: e.clientX, y: e.clientY }}; }});
            renderer.domElement.addEventListener('mousemove', (e) => {{ if (isMouseDown && previousMouse) {{ const deltaX = e.clientX - previousMouse.x; const deltaY = e.clientY - previousMouse.y; rotationY += deltaX * 0.008; rotationX += deltaY * 0.008; rotationX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotationX)); updateRotation(); previousMouse = {{ x: e.clientX, y: e.clientY }}; }} }});
            renderer.domElement.addEventListener('mouseup', () => {{ isMouseDown = false; previousMouse = null; }});
            renderer.domElement.addEventListener('wheel', (e) => {{ e.preventDefault(); camera.position.z += e.deltaY * 0.001; camera.position.z = Math.max(1, Math.min(10, camera.position.z)); }});
        }}
        function updateRotation() {{
            if (meshCorps) {{ meshCorps.rotation.order = 'YXZ'; meshCorps.rotation.y = rotationY; meshCorps.rotation.x = rotationX; }}
            if (meshVetement) {{ meshVetement.rotation.order = 'YXZ'; meshVetement.rotation.y = rotationY; meshVetement.rotation.x = rotationX; }}
        }}
        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        function onWindowResize() {{ camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }}
        init();
    </script>
</body>
</html>
    """
    return html_template


@app.route('/api/vetement/preview/<vetement_id>', methods=['GET'])
def get_vetement_preview(vetement_id):
    try:
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Vêtement non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices_corps = data['vertices_corps']
        vertices_avec_vetement = data['vertices_avec_vetement']
        faces = data['faces']
        masque_vetement = data['masque_vetement']
        couleur = str(data['couleur'])
        type_vetement = str(data['type_vetement'])
        
        html_content = generer_html_vetement_preview(
            vertices_corps, vertices_avec_vetement, faces,
            masque_vetement, couleur, type_vetement
        )
        
        html_file = os.path.join(PREVIEW_DIR, f"{vetement_id}_preview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        webview_url = f"{BASE_URL}/webview/{vetement_id}_preview.html"
        
        return jsonify({
            'success': True,
            'preview_url': webview_url,
            'type': 'html_three_js',
            'message': f'Aperçu {type_vetement} {couleur} prêt'
        })
        
    except Exception as e:
        print(f"❌ Erreur preview vêtement: {e}")
        return jsonify({'error': str(e)}), 500


def generate_fallback_preview(vetement_id, type_vetement, couleur, vertices):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        x = vertices[:, 0]; y = vertices[:, 1]; z = vertices[:, 2]
        couleur_rgb = COULEURS_DISPONIBLES.get(couleur, [128, 128, 128])
        couleur_normalized = [c/255.0 for c in couleur_rgb]
        
        ax.scatter(x, y, z, c=[couleur_normalized], s=1, alpha=0.8)
        ax.set_title(f"{type_vetement} {couleur}", fontsize=14, pad=20)
        ax.view_init(elev=20, azim=45)
        
        preview_file = os.path.join(PREVIEW_DIR, f"{vetement_id}_fallback_preview.png")
        plt.savefig(preview_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return send_file(preview_file, mimetype='image/png', as_attachment=False)
        
    except Exception as e:
        return jsonify({'error': 'Impossible de générer le preview', 'details': str(e)}), 500


@app.route('/api/vetement/visualize/<vetement_id>', methods=['POST'])
def visualize_vetement_3d(vetement_id):
    try:
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Vêtement non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices_corps = data['vertices_corps']
        vertices_avec_vetement = data['vertices_avec_vetement']
        faces = data['faces']
        masque_vetement = data['masque_vetement']
        couleur = str(data['couleur'])
        type_vetement = str(data['type_vetement'])
        
        request_data = request.get_json() or {}
        angle_camera = request_data.get('angle_camera', 0)
        distance_camera = request_data.get('distance_camera', 3.0)
        hauteur_camera = request_data.get('hauteur_camera', 0.0)
        
        try:
            import vedo
            
            mesh_corps = vedo.Mesh([vertices_corps, faces])
            mesh_corps.color('lightblue').alpha(0.7)
            
            mesh_vetement_data = VetementGenerator.creer_mesh_jupe_separe(
                vertices_avec_vetement, masque_vetement, couleur
            )
            
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data['mesh_object'] is not None:
                meshes_a_afficher.append(mesh_vetement_data['mesh_object'])
            
            plotter = vedo.Plotter(offscreen=True, size=(800, 600))
            for mesh in meshes_a_afficher:
                plotter.add(mesh)
            
            bounds = meshes_a_afficher[0].bounds()
            center_y = (bounds[2] + bounds[3]) / 2
            camera_height = bounds[2] + (bounds[3] - bounds[2]) * 0.3
            camera_x = distance_camera * np.cos(np.radians(angle_camera))
            camera_z = distance_camera * np.sin(np.radians(angle_camera))
            
            plotter.camera.SetPosition(camera_x, camera_height + hauteur_camera, camera_z)
            plotter.camera.SetFocalPoint(0, center_y, 0)
            plotter.camera.SetViewUp(0, 1, 0)
            
            screenshot_path = os.path.join(TEMP_DIR, f"{vetement_id}_view_{angle_camera}.png")
            plotter.screenshot(screenshot_path)
            plotter.close()
            
            return jsonify({
                'success': True,
                'message': f'Visualisation {type_vetement} générée',
                'screenshot_path': screenshot_path,
                'info': {
                    'vetement_id': vetement_id,
                    'type_vetement': type_vetement,
                    'couleur': couleur,
                    'angle_camera': angle_camera,
                    'mesh_separe_utilise': mesh_vetement_data['mesh_object'] is not None,
                }
            })
            
        except ImportError:
            return jsonify({'error': 'Vedo non disponible'}), 500
        except Exception as e:
            return jsonify({'error': f'Erreur visualisation: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mannequin/generate', methods=['POST'])
def generate_mannequin():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données JSON requises'}), 400
        
        gender = data.get('gender', 'neutral')
        mesures = data.get('mesures', {})
        
        genders_valides = ['neutral', 'male', 'female']
        if gender not in genders_valides:
            return jsonify({
                'error': f'Genre non valide: {gender}',
                'genders_disponibles': genders_valides
            }), 400
        
        mesures_default = {
            'tour_taille': 68, 'tour_hanches': 92,
            'tour_poitrine': 88, 'hauteur': 170, 'longueur_bras': 60
        }
        mesures = {**mesures_default, **mesures}
        
        try:
            if not mannequin_gen.charger_modele_star(gender):
                return jsonify({'error': 'Impossible de charger le modèle STAR'}), 500
        except Exception as e:
            return jsonify({'error': f'Erreur chargement STAR: {str(e)}'}), 500
        
        vertices_base = mannequin_gen.v_template.copy()
        faces = mannequin_gen.f
        
        if mannequin_gen.shapedirs is not None:
            try:
                joints_base = mannequin_gen.J_regressor.dot(vertices_base)
                mesures_actuelles = mannequin_gen.calculer_mesures_modele(vertices_base, joints_base, DEFAULT_MAPPING)
                vertices_final, betas = mannequin_gen.deformer_modele(mesures, mesures_actuelles)
            except Exception as e:
                print(f"⚠️ Erreur déformation: {e}")
                vertices_final = vertices_base
                betas = np.zeros(10)
        else:
            vertices_final = vertices_base
            betas = np.zeros(10)
        
        import random, string, time as time_module
        timestamp = str(int(time_module.time()))[-6:]
        random_suffix = ''.join(random.choices(string.digits, k=3))
        mannequin_id = f"mannequin_{gender}_{timestamp}_{random_suffix}"
        
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        try:
            np.savez_compressed(
                temp_file,
                vertices=vertices_final,
                faces=faces,
                gender=gender,
                mesures=mesures,
                betas=betas,
                joints=mannequin_gen.J_regressor.dot(vertices_final) if mannequin_gen.J_regressor is not None else np.array([])
            )
        except Exception as e:
            return jsonify({'error': f'Erreur sauvegarde: {str(e)}'}), 500
        
        hauteur, y_min, y_max = visualisateur.calculer_hauteur_mannequin(vertices_final)
        
        info = {
            'gender': gender,
            'vertices_count': len(vertices_final),
            'faces_count': len(faces),
            'joints_count': len(mannequin_gen.Jtr) if mannequin_gen.Jtr is not None else 0,
            'betas': betas.tolist(),
            'mesures_appliquees': mesures,
            'has_shapedirs': mannequin_gen.shapedirs is not None,
            'dimensions': {
                'hauteur_totale': round(hauteur, 2),
                'y_min_pieds': round(y_min, 2),
                'y_max_tete': round(y_max, 2),
                'centre_reel': round((y_min + y_max) / 2, 2)
            }
        }
        
        info_convertie = convertir_numpy_pour_json(info)
        
        return jsonify({
            'success': True,
            'message': f'✅ Mannequin {gender} généré',
            'info': info_convertie,
            'mannequin_id': mannequin_id
        })
        
    except Exception as e:
        print(f"❌ Erreur génération mannequin: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generer_html_mannequin_preview(vertices, faces, gender):
    vertices_json = json.dumps(vertices.tolist())
    faces_json = json.dumps(faces.tolist())
    
    html_template = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mannequin {gender} - Aperçu</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{ width: 100vw; height: 100vh; overflow: hidden; font-family: Arial, sans-serif; background: #f5f5f5; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 12px; border-radius: 8px; font-size: 13px; z-index: 10; }}
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.8); color: white; padding: 20px; border-radius: 8px; text-align: center; z-index: 100; }}
    </style>
</head>
<body>
    <div id="loading">Chargement du mannequin...</div>
    <div id="info"><strong>Mannequin {gender}</strong><br>Vertices: {len(vertices)}<br>Glissez pour tourner</div>
    <script>
        let scene, camera, renderer, mannequin;
        let rotationX = 0, rotationY = 0;
        const vertices = {vertices_json};
        const faces = {faces_json};
        function init() {{
            scene = new THREE.Scene(); scene.background = new THREE.Color(0xfafafa);
            const yValues = vertices.map(v => v[1]);
            const yMin = Math.min(...yValues); const yMax = Math.max(...yValues);
            const hauteur = yMax - yMin; const centreY = (yMin + yMax) / 2;
            camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
            const dist = Math.max(2, hauteur * 2.2);
            camera.position.set(0, centreY * 0.8, dist);
            camera.lookAt(0, centreY, 0);
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices.flat()), 3));
            geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.flat()), 1));
            geometry.computeVertexNormals();
            mannequin = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ color: 0xd4a574, shininess: 20, side: THREE.DoubleSide }}));
            scene.add(mannequin);
            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8); directionalLight.position.set(5, 10, 5); scene.add(directionalLight);
            document.addEventListener('mousedown', (e) => {{ mouseDown = true; prevX = e.clientX; prevY = e.clientY; }});
            document.addEventListener('mousemove', (e) => {{ if (!mouseDown) return; rotationY += (e.clientX - prevX) * 0.005; rotationX += (e.clientY - prevY) * 0.005; mannequin.rotation.order = 'YXZ'; mannequin.rotation.y = rotationY; mannequin.rotation.x = rotationX; prevX = e.clientX; prevY = e.clientY; }});
            document.addEventListener('mouseup', () => {{ mouseDown = false; }});
            let touchStart = null;
            document.addEventListener('touchstart', (e) => {{ if (e.touches.length === 1) touchStart = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }});
            document.addEventListener('touchmove', (e) => {{ if (!touchStart || e.touches.length !== 1) return; rotationY += (e.touches[0].clientX - touchStart.x) * 0.005; rotationX += (e.touches[0].clientY - touchStart.y) * 0.005; mannequin.rotation.order = 'YXZ'; mannequin.rotation.y = rotationY; mannequin.rotation.x = rotationX; touchStart = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }}; }});
            document.addEventListener('touchend', () => {{ touchStart = null; }});
            document.getElementById('loading').style.display = 'none';
            animate();
            window.addEventListener('resize', () => {{ camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }});
        }}
        let mouseDown = false, prevX = 0, prevY = 0;
        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        init();
    </script>
</body>
</html>
    """
    return html_template


@app.route('/api/mannequin/preview/<mannequin_id>', methods=['GET'])
def get_mannequin_preview(mannequin_id):
    try:
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        gender = str(data['gender'])
        
        html_content = generer_html_mannequin_preview(vertices, faces, gender)
        
        html_file = os.path.join(PREVIEW_DIR, f"{mannequin_id}_preview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        webview_url = f"{BASE_URL}/webview/{mannequin_id}_preview.html"
        
        return jsonify({
            'success': True,
            'preview_url': webview_url,
            'type': 'html_three_js',
            'message': f'Aperçu mannequin {gender} prêt'
        })
        
    except Exception as e:
        print(f"❌ Erreur preview mannequin: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vetement/types', methods=['GET'])
def get_vetement_types():
    types_detailles = {}
    for nom, params in TYPES_VETEMENTS.items():
        types_detailles[nom] = {
            'description': params['description'],
            'categorie': params['categorie'],
            'type': params['type'],
            'longueur_relative': params['longueur_relative']
        }
    
    return jsonify({
        'types_vetements': types_detailles,
        'couleurs': COULEURS_DISPONIBLES,
        'corrections_jupes': {
            'jupe_ovale': '✅ Reste ample jusqu\'en bas (rayon_bas = rayon_max)',
            'jupe_droite': '✅ Profil uniforme sans évasement excessif',
        }
    })


@app.route('/api/mannequin/visualize/<mannequin_id>', methods=['POST'])
def visualize_mannequin_3d(mannequin_id):
    try:
        if not VEDO_AVAILABLE:
            return jsonify({'error': 'Vedo non disponible'}), 400
        
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        gender = str(data['gender'])
        
        def visualiser_mannequin():
            try:
                from vedo import Mesh, Plotter
                mesh_mannequin = Mesh([vertices, faces])
                mesh_mannequin.color([0.8, 0.6, 0.4]).alpha(0.9)
                plt = Plotter(bg='white', axes=1, title=f"Mannequin {gender}", interactive=True)
                plt.add(mesh_mannequin)
                visualisateur.configurer_camera_pieds_visibles(plt, vertices)
                plt.show(interactive=True)
                plt.close()
            except Exception as e:
                print(f"❌ Erreur thread visualisation: {e}")
        
        response_data = {
            'success': True,
            'message': f'✅ Visualisation 3D mannequin {gender} lancée',
            'mannequin_id': mannequin_id,
            'gender': gender
        }
        
        thread = threading.Thread(target=visualiser_mannequin)
        thread.daemon = True
        thread.start()
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Erreur visualisation 3D mannequin: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Démarrage serveur Flask")
    print(f"🌍 Environnement: {'PRODUCTION (Render)' if IS_PRODUCTION else 'DÉVELOPPEMENT (Local)'}")
    print(f"🔗 Base URL: {BASE_URL}")
    print(f"🔌 Port: {port}")
    print(f"✅ Corrections jupes:")
    print(f"   - Jupe ovale: rayon_bas = rayon_max (plus de rentrée vers l'intérieur)")
    print(f"   - Jupe droite: rayon_bas = rayon_hanches × 1.12 (profil uniforme)")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=not IS_PRODUCTION
    )
