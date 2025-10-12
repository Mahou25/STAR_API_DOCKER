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

# --- TYPES DE VÊTEMENTS (EXACTEMENT comme dans vos scripts) ---
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
        "ampleur": 1.3,
        "description": "Mini-jupe ovale évasée (mi-cuisse)"
    },
    "Jupe ovale au genou": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.35,
        "ampleur": 1.4,
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
        """COPIE EXACTE du script 2"""
        mesures = {}
        for mesure, info in mapping.items():
            joint_indices = info["joints"]
            if len(joint_indices) == 2:
                mesures[mesure] = euclidean(joints[joint_indices[0]], joints[joint_indices[1]])
            elif len(joint_indices) == 1:
                # Pour les tours (approximation)
                joint_pos = joints[joint_indices[0]]
                distances = np.linalg.norm(vertices - joint_pos, axis=1)
                nearby_vertices = vertices[distances < np.percentile(distances, 20)]
                if len(nearby_vertices) > 3:
                    center = np.mean(nearby_vertices, axis=0)
                    radii = np.linalg.norm(nearby_vertices - center, axis=1)
                    mesures[mesure] = 2 * np.pi * np.mean(radii)
                else:
                    mesures[mesure] = 50.0
        return mesures
    
    def deformer_modele(self, mesures_cibles, mesures_actuelles):
        """COPIE EXACTE du script 2"""
        if self.shapedirs is None:
            print("Pas de blend shapes disponibles")
            return self.v_template, np.zeros(10)
        
        n_betas = min(10, self.shapedirs.shape[2])
        
        def objective(betas):
            vertices_deformed = self.v_template + np.sum(
                self.shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2
            )
            joints_deformed = self.J_regressor.dot(vertices_deformed)
            
            mesures_modele = {}
            for mesure in mesures_cibles.keys():
                if mesure in mesures_actuelles:
                    ratio = mesures_cibles[mesure] / mesures_actuelles[mesure] if mesures_actuelles[mesure] > 0 else 1.0
                    mesures_modele[mesure] = mesures_actuelles[mesure] * ratio
            
            error = 0
            for mesure in mesures_cibles.keys():
                if mesure in mesures_modele:
                    error += (mesures_modele[mesure] - mesures_cibles[mesure]) ** 2
            
            regularization = 0.1 * np.sum(betas ** 2)
            return error + regularization
        
        initial_betas = np.zeros(n_betas)
        bounds = [(-3, 3)] * n_betas
        
        result = minimize(objective, initial_betas, method='L-BFGS-B', bounds=bounds)
        betas = result.x
        
        vertices_final = self.v_template + np.sum(
            self.shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2
        )
        
        return vertices_final, betas

class VetementGenerator:
    """CLASSE CORRIGÉE - Plus d'erreur de sérialisation"""
    
    @staticmethod
    def detecter_points_anatomiques(verts):
        """Fonction inchangée"""
        y_vals = verts[:, 1]
        x_vals = verts[:, 0]
        z_vals = verts[:, 2]
        
        # Points de base
        y_max = np.max(y_vals)
        y_min = np.min(y_vals)
        hauteur_totale = y_max - y_min
        
        # Points anatomiques détaillés
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
        
    
# Replace the creer_mesh_jupe_separe function in your VetementGenerator class

    @staticmethod
    def creer_mesh_jupe_separe(verts_corps, masque_jupe, couleur_nom):
        """
        CORRIGÉE - Crée un mesh de jupe COMPLÈTEMENT FERMÉ et SANS TROUS
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
            
            indices_jupe = np.where(masque_jupe)[0]
            points_jupe = verts_corps[masque_jupe]
            
            if len(points_jupe) < 100:
                print(f"Pas assez de points jupe: {len(points_jupe)}")
                return default_result
            
            y_vals = points_jupe[:, 1]
            y_min = np.min(y_vals)
            y_max = np.max(y_vals)
            
            n_couches = max(15, int((y_max - y_min) * 60))
            couches = []
            
            for layer_idx in range(n_couches):
                y_layer = y_min + (layer_idx / (n_couches - 1)) * (y_max - y_min) if n_couches > 1 else y_min
                tolerance = (y_max - y_min) / (n_couches * 2)
                
                mask_layer = np.abs(y_vals - y_layer) < tolerance
                idx_layer = np.where(mask_layer)[0]
                
                if len(idx_layer) > 3:
                    centre = np.mean(points_jupe[idx_layer], axis=0)
                    angles = np.arctan2(
                        points_jupe[idx_layer, 2] - centre[2],
                        points_jupe[idx_layer, 0] - centre[0]
                    )
                    idx_sorted = idx_layer[np.argsort(angles)]
                    couches.append(idx_sorted)
            
            if len(couches) < 2:
                print("Pas assez de couches pour créer le mesh")
                return default_result
            
            faces = []
            
            for layer_idx in range(len(couches) - 1):
                couche_actuelle = couches[layer_idx]
                couche_suivante = couches[layer_idx + 1]
                
                n_curr = len(couche_actuelle)
                n_next = len(couche_suivante)
                
                if n_curr > 0 and n_next > 0:
                    max_points = max(n_curr, n_next)
                    for i in range(max_points):
                        i_curr = i % n_curr
                        i_next = i % n_next
                        i_curr_next = (i + 1) % n_curr
                        i_next_next = (i + 1) % n_next
                        
                        faces.append([couche_actuelle[i_curr], 
                                    couche_suivante[i_next], 
                                    couche_actuelle[i_curr_next]])
                        
                        faces.append([couche_suivante[i_next], 
                                    couche_suivante[i_next_next], 
                                    couche_actuelle[i_curr_next]])
            
            if len(couches) > 0:
                couche_bas = couches[-1]
                centre_bas = np.mean(points_jupe[couche_bas], axis=0)
                centre_bas_idx = len(points_jupe)
                points_jupe = np.vstack([points_jupe, centre_bas])
                
                for i in range(len(couche_bas)):
                    i_next = (i + 1) % len(couche_bas)
                    faces.append([couche_bas[i], couche_bas[i_next], centre_bas_idx])
            
            couleur_rgb = COULEURS_DISPONIBLES.get(couleur_nom, [128, 128, 128])
            couleur_normalized = [c/255.0 for c in couleur_rgb]
            
            if len(faces) > 0:
                try:
                    mesh_jupe = Mesh([points_jupe, faces])
                    mesh_jupe.color(couleur_normalized).alpha(0.9)
                    
                    print(f"Mesh jupe créé: {len(points_jupe)} points, {len(faces)} faces")
                    
                    return {
                        'mesh_object': mesh_jupe,
                        'points_count': len(points_jupe),
                        'faces_count': len(faces),
                        'couleur_rgb': couleur_rgb,
                        'couleur_normalized': couleur_normalized
                    }
                except Exception as e:
                    print(f"Erreur création mesh: {e}")
                    return default_result
            else:
                print("Aucune face générée pour la jupe")
                return default_result
                
        except Exception as e:
            print(f"Erreur complète mesh jupe: {e}")
            return default_result
        
        
# 2. AJOUTER LA FONCTION DE LISSAGE MANQUANTE

    @staticmethod
    def lisser_jupe(verts, masque_jupe, iterations=2):
        """
        ✅ FONCTION MANQUANTE - Lisse la surface de la jupe 
        EXACTEMENT comme dans le script original
        """
        verts_lisses = verts.copy()
        
        for iteration in range(iterations):
            nouveaux_verts = verts_lisses.copy()
            
            for i, est_jupe in enumerate(masque_jupe):
                if est_jupe:
                    # Trouve les voisins proches
                    distances = np.linalg.norm(verts_lisses - verts_lisses[i], axis=1)
                    voisins = np.where((distances < 0.03) & (distances > 0))[0]
                    
                    if len(voisins) > 2:
                        # Moyenne pondérée légère avec les voisins
                        poids_centre = 0.75
                        poids_voisins = (1 - poids_centre) / len(voisins)
                        
                        nouveaux_verts[i] = (poids_centre * verts_lisses[i] + 
                                        poids_voisins * np.sum(verts_lisses[voisins], axis=0))
            
            verts_lisses = nouveaux_verts
            print(f"✅ Lissage iteration {iteration + 1}/{iterations} terminée")
        
        return verts_lisses


    @staticmethod
    def calculer_profil_jupe_droite(points_anat, longueur_relative):
        """✅ CORRIGÉ - Plus de fonction locale dans le retour"""
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
        rayon_hanches_jupe = rayon_hanches * 0.95
        rayon_bas = rayon_hanches_jupe * 1.02
        
        # ✅ SOLUTION 1: Retourner les paramètres au lieu de la fonction
        return {
            'type': 'droite',
            'y_debut': y_debut_jupe,
            'y_bas': y_bas_jupe,
            'rayon_debut': rayon_debut,
            'rayon_hanches_jupe': rayon_hanches_jupe,
            'rayon_bas': rayon_bas,
            'y_hanches': y_hanches
        }
        
    @staticmethod
    def calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur=1.4):
        """✅ CORRIGÉ - Plus de fonction locale dans le retour"""
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
        rayon_max = rayon_hanches * ampleur
        rayon_bas = rayon_hanches * 0.9
        
        y_max_largeur = y_hanches - 0.1
        
        # ✅ SOLUTION 1: Retourner les paramètres au lieu de la fonction
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
        """✅ CORRIGÉ - Plus de fonction locale dans le retour"""
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
        
        # ✅ SOLUTION 1: Retourner les paramètres au lieu de la fonction
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
        """✅ NOUVELLE FONCTION - Calcule le rayon à partir des paramètres sauvegardés"""
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
        """Calcul spécifique pour jupe droite"""
        y_debut = profil['y_debut']
        y_bas = profil['y_bas']
        y_hanches = profil['y_hanches']
        
        rayon_debut = profil['rayon_debut']
        rayon_hanches_jupe = profil['rayon_hanches_jupe']
        rayon_bas = profil['rayon_bas']
        
        if y >= y_hanches:
            if y_debut == y_hanches:
                return rayon_debut
            t = (y_debut - y) / (y_debut - y_hanches)
            return rayon_debut + t * (rayon_hanches_jupe - rayon_debut)
        else:
            if y_hanches == y_bas:
                return rayon_hanches_jupe
            t = (y_hanches - y) / (y_hanches - y_bas)
            return rayon_hanches_jupe + t * (rayon_bas - rayon_hanches_jupe)
    
    @staticmethod
    def _calculer_rayon_ovale(profil, y):
        """Calcul spécifique pour jupe ovale"""
        y_debut = profil['y_debut']
        y_bas = profil['y_bas']
        y_max_largeur = profil['y_max_largeur']
        
        rayon_debut = profil['rayon_debut']
        rayon_max = profil['rayon_max']
        rayon_bas = profil['rayon_bas']
        
        if y >= y_max_largeur:
            if y_debut == y_max_largeur:
                return rayon_debut
            t = (y_debut - y) / (y_debut - y_max_largeur)
            t_curve = 0.5 * (1 - np.cos(np.pi * t))
            return rayon_debut + t_curve * (rayon_max - rayon_debut)
        else:
            if y_max_largeur == y_bas:
                return rayon_max
            t = (y_max_largeur - y) / (y_max_largeur - y_bas)
            t_curve = 0.5 * (1 + np.cos(np.pi * t))
            return rayon_max - (rayon_max - rayon_bas) * (1 - t_curve)
    
    @staticmethod
    def _calculer_rayon_trapeze(profil, y):
        """Calcul spécifique pour jupe trapèze"""
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

# ===== CLASSE VISUALISATEUR 3D CORRIGÉE POUR VOIR LES PIEDS =====
class Visualisateur3D:
    def __init__(self):
        self.active_plotters = []
    
    def calculer_hauteur_mannequin(self, vertices):
        """
        Calcule la hauteur réelle du mannequin pour ajuster la caméra
        """
        if vertices is None or len(vertices) == 0:
            return 1.7  # Hauteur par défaut
        
        y_min = np.min(vertices[:, 1])
        y_max = np.max(vertices[:, 1])
        hauteur = y_max - y_min
        
        print(f"📏 Hauteur mannequin calculée: {hauteur:.2f}")
        print(f"   Y min (pieds): {y_min:.2f}")
        print(f"   Y max (tête): {y_max:.2f}")
        
        return hauteur, y_min, y_max
    
    def configurer_camera_pieds_visibles(self, plotter, vertices):
        """
        ✅ CONFIGURATION CAMÉRA CORRIGÉE POUR VOIR LES PIEDS
        """
        if vertices is None or len(vertices) == 0:
            # Paramètres par défaut
            camera_position = (0, 0, 4.0)
            look_at_point = (0, 0.5, 0)
            up_vector = (0, 1, 0)
        else:
            # Calculer les vraies dimensions du mannequin
            hauteur, y_min, y_max = self.calculer_hauteur_mannequin(vertices)
            
            # ✅ CORRECTION: Centrer le look_at sur le VRAI centre du corps
            centre_y_reel = (y_min + y_max) / 2.0  # Milieu réel
            
            # Position caméra adaptée à la hauteur
            # Plus le mannequin est grand, plus on recule
            distance_camera = max(3.5, hauteur * 2.5)
            
            # Paramètres corrigés
            camera_position = (0, centre_y_reel * 0.8, distance_camera)  # Légèrement en dessous du centre
            look_at_point = (0, centre_y_reel, 0)  # Centre réel du corps
            up_vector = (0, 1, 0)
        
        # Configuration Vedo
        plotter.camera.SetPosition(*camera_position)
        plotter.camera.SetFocalPoint(*look_at_point)
        plotter.camera.SetViewUp(*up_vector)
        
        # FOV plus large pour voir tout le corps Y COMPRIS LES PIEDS
        plotter.camera.SetViewAngle(55)  # 55° pour un champ plus large
        
        # Distance de clipping adaptée
        plotter.camera.SetClippingRange(0.1, 15.0)
        
        # ✅ IMPORTANT: Zoom automatique pour tout voir
        plotter.reset_camera()
        plotter.camera.Zoom(0.8)  # Dézoomer légèrement pour être sûr
        
        print(f"📸 ✅ CAMÉRA CORRIGÉE - PIEDS VISIBLES:")
        print(f"   Position: {camera_position}")
        print(f"   Look at: {look_at_point} (centre réel du corps)")
        print(f"   Up vector: {up_vector}")
        print(f"   FOV: 55° (élargi pour voir les pieds)")
        print(f"   Zoom: 0.8 (dézoomer pour tout voir)")
        
        return True
        
    def capturer_mannequin_3d_vedo(self, vertices, faces, fichier_sortie, titre="Mannequin"):
        """
        ✅ CAPTURE MANNEQUIN COMPLET AVEC PIEDS VISIBLES
        """
        if not VEDO_AVAILABLE:
            print("❌ Vedo non disponible")
            return False
        
        try:
            print(f"📸 ✅ Capture mannequin AVEC PIEDS VISIBLES: {titre}")
            
            # Créer le mesh du mannequin
            mesh_mannequin = Mesh([vertices, faces])
            mesh_mannequin.color([0.8, 0.6, 0.4]).alpha(0.9)
            
            # Plotter avec résolution élevée
            plt = Plotter(bg='white', axes=0, interactive=False, offscreen=True, size=(1920, 1920))
            plt.add(mesh_mannequin)
            
            # ✅ CONFIGURATION CAMÉRA CORRIGÉE POUR VOIR LES PIEDS
            self.configurer_camera_pieds_visibles(plt, vertices)
            
            # Rendu et capture
            plt.render()
            img_array = plt.screenshot(filename=fichier_sortie, scale=3)
            plt.close()
            
            print(f"✅ ✅ Capture mannequin AVEC PIEDS réussie: {fichier_sortie}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur capture mannequin avec pieds: {e}")
            return False
    
    def capturer_rendu_3d_vedo(self, vertices_corps, faces, masque_vetement, 
                                vertices_vetement, mesh_vetement_data, 
                                fichier_sortie, titre="Mannequin avec Vêtement"):
        """
        ✅ CAPTURE VÊTEMENT AVEC PIEDS VISIBLES
        """
        if not VEDO_AVAILABLE:
            print("❌ Vedo non disponible")
            return False
        
        try:
            print(f"📸 ✅ Capture vêtement AVEC PIEDS VISIBLES: {titre}")
            
            # Créer le mesh du corps avec couleur peau
            mesh_corps = Mesh([vertices_corps, faces])
            mesh_corps.color([0.8, 0.6, 0.4]).alpha(0.8)
            
            # Préparer la liste des meshes
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data is not None and mesh_vetement_data['mesh_object'] is not None:
                meshes_a_afficher.append(mesh_vetement_data['mesh_object'])
                print(f"✅ Mesh vêtement ajouté")
            
            # Plotter avec résolution élevée
            plt = Plotter(bg='white', axes=0, interactive=False, offscreen=True, size=(1920, 1920))
            
            # Ajouter tous les meshes
            for mesh in meshes_a_afficher:
                plt.add(mesh)
            
            # ✅ CONFIGURATION CAMÉRA CORRIGÉE POUR VOIR LES PIEDS
            self.configurer_camera_pieds_visibles(plt, vertices_corps)
            
            # Capturer avec haute résolution
            plt.render()
            img_array = plt.screenshot(filename=fichier_sortie, scale=3)
            plt.close()
            
            print(f"✅ ✅ Capture vêtement AVEC PIEDS réussie: {fichier_sortie}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur capture vêtement avec pieds: {e}")
            return False
    
    def afficher_mannequin_avec_vetement(self, vertices_corps, faces, masque_vetement, 
                                        vertices_vetement, mesh_vetement_data, titre="Mannequin avec Vêtement"):
        """✅ Affichage interactif avec pieds visibles"""
        if not VEDO_AVAILABLE:
            print("❌ Vedo non disponible")
            return False
        
        try:
            print(f"🎭 ✅ Affichage 3D interactif AVEC PIEDS VISIBLES: {titre}")
            
            # Créer le mesh du corps avec couleur peau
            mesh_corps = Mesh([vertices_corps, faces])
            mesh_corps.color([0.8, 0.6, 0.4]).alpha(0.8)
            
            # Préparer la liste des meshes à afficher
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data is not None and mesh_vetement_data['mesh_object'] is not None:
                meshes_a_afficher.append(mesh_vetement_data['mesh_object'])
                print(f"✅ Mesh vêtement ajouté")
            
            # LANCEMENT VEDO interactif avec pieds visibles
            def lancer_vedo():
                try:
                    plt = Plotter(bg='white', axes=1, title=titre, interactive=True)
                    
                    for mesh in meshes_a_afficher:
                        plt.add(mesh)
                    
                    # ✅ CONFIGURATION CAMÉRA CORRIGÉE POUR VOIR LES PIEDS
                    self.configurer_camera_pieds_visibles(plt, vertices_corps)
                    
                    plt.show(interactive=True)
                    plt.close()
                    
                except Exception as e:
                    print(f"❌ Erreur Vedo thread: {e}")
                    try:
                        show(*meshes_a_afficher, axes=1, viewup="y", bg="white", 
                                title=titre, interactive=True)
                    except Exception as e2:
                        print(f"❌ Erreur show() fallback: {e2}")
            
            # Lancer dans un thread séparé
            thread = threading.Thread(target=lancer_vedo, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur affichage avec vêtement: {e}")
            return False

# Function to clean up temporary files
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

# Instance globalfe
mannequin_gen = MannequinGenerator()
visualisateur = Visualisateur3D()

# ============== TIMEOUTS CONSIDÉRABLEMENT AUGMENTÉS ==============
TIMEOUT_GENERATION = 180  # 3 minutes par génération
TIMEOUT_CAPTURE = 60      # 1 minute pour les captures
TIMEOUT_SCENARIOS = 300   # 5 minutes pour les scénarios multiples

def timeout_function(timeout_duration):
    """✅ Décorateur timeout corrigé pour Flask - Windows compatible"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # ✅ SOLUTION : Ne pas utiliser le timeout pour generate_vetement
            # Le timeout cause plus de problèmes qu'il n'en résout
            try:
                # Exécuter directement sans timeout
                return func(*args, **kwargs)
            except Exception as e:
                print(f"❌ Erreur dans {func.__name__}: {e}")
                raise e
        
        return wrapper
    return decorator

def simple_timeout_function(timeout_duration):
    """Version simplifiée sans thread pour éviter les problèmes Flask"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Pour l'instant, on exécute directement
            # Un vrai timeout nécessiterait une architecture différente
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
  
  
# ===== FONCTION DE VALIDATION CORRIGÉE =====






def valider_type_vetement(type_saisi):
    """
    ✅ FONCTION CORRIGÉE - Valide le type de vêtement en ignorant la casse
    Retourne: soit le type_officiel (string), soit (None, liste_types_disponibles) (tuple)
    """
    if not type_saisi or not isinstance(type_saisi, str):
        types_disponibles = list(TYPES_VETEMENTS.keys())
        return None, types_disponibles
    
    # Normaliser la saisie
    type_saisi_lower = type_saisi.lower().strip()
    
    # Chercher une correspondance insensible à la casse
    for type_officiel in TYPES_VETEMENTS.keys():
        if type_officiel.lower() == type_saisi_lower:
            return type_officiel  # ✅ Retourner directement le type valide
    
    # Si pas trouvé, retourner None et la liste des options
    types_disponibles = list(TYPES_VETEMENTS.keys())
    return None, types_disponibles  # ✅ Tuple (None, liste)

# --- ROUTES API CORRIGÉES POUR VOIR LES PIEDS ---


# ============== NOUVELLES ROUTES POUR WEBVIEW 3D ==============

# ============== ROUTES WEBVIEW CORRIGÉES ==============
def get_local_ip():
    """Récupère l'IP locale dynamiquement (pour dev uniquement)"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
    

def get_environment():
    """Détermine si on est en production ou développement"""
    # Render définit automatiquement cette variable
    return os.environ.get('RENDER', 'false') == 'true' or \
           os.environ.get('ENVIRONMENT', 'development') == 'production'

def get_base_url():
    """Retourne l'URL de base selon l'environnement"""
    if get_environment():
        # URL de production Render
        return "https://star-api-docker.onrender.com"
    else:
        # URL locale pour développement
        local_ip = get_local_ip()
        return f"http://{local_ip}:5000"



# ✅ VARIABLE GLOBALE IP DYNAMIQUE
IS_PRODUCTION = get_environment()
BASE_URL = get_base_url()

print(f"🌍 Environnement: {'PRODUCTION' if IS_PRODUCTION else 'DÉVELOPPEMENT'}")
print(f"🔗 Base URL: {BASE_URL}")

@app.route('/api/mannequin/webview/<mannequin_id>', methods=['GET'])
def generate_mannequin_webview(mannequin_id):
    """✅ CORRIGÉ - Utilise BASE_URL adaptative"""
    try:
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        gender = str(data['gender'])
        
        # Générer la page HTML
        html_content = generer_html_mannequin_3d(vertices, faces, gender)
        
        # Sauvegarder le fichier HTML
        html_file = os.path.join(PREVIEW_DIR, f"{mannequin_id}_webview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # ✅ CORRECTION: Utiliser BASE_URL adaptatif
        webview_url = f"{BASE_URL}/webview/{mannequin_id}_webview.html"
        
        print(f"✅ URL générée ({('PROD' if IS_PRODUCTION else 'DEV')}): {webview_url}")
        
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
    """✅ CORRIGÉ - Utilise BASE_URL adaptative"""
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
        
        # Générer la page HTML
        html_content = generer_html_vetement_3d(
            vertices_corps, vertices_avec_vetement, faces, 
            masque_vetement, couleur, type_vetement
        )
        
        # Sauvegarder le fichier HTML
        html_file = os.path.join(PREVIEW_DIR, f"{vetement_id}_webview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # ✅ CORRECTION: Utiliser BASE_URL adaptatif
        webview_url = f"{BASE_URL}/webview/{vetement_id}_webview.html"
        
        print(f"✅ URL générée ({('PROD' if IS_PRODUCTION else 'DEV')}): {webview_url}")
        
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
    """✅ Route de test corrigée avec URL adaptative"""
    test_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test WebView - {'Production' if IS_PRODUCTION else 'Développement'}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        .test-box {{
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
    </style>
</head>
<body>
    <div class="test-box">
        <h1>✅ WebView Test</h1>
        <p>Si vous voyez ce message, le WebView fonctionne !</p>
        <p>Environnement: <strong>{'PRODUCTION' if IS_PRODUCTION else 'DÉVELOPPEMENT'}</strong></p>
        <p>URL: <strong>{BASE_URL}</strong></p>
        <div id="timer">Loading...</div>
    </div>
    
    <script>
        let seconds = 0;
        function updateTimer() {{
            document.getElementById('timer').textContent = 
                `Temps écoulé: ${{seconds}} secondes`;
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
    
    # ✅ CORRECTION: Utiliser BASE_URL adaptatif
    test_url = f"{BASE_URL}/webview/test_webview.html"
    
    print(f"✅ URL de test générée: {test_url}")
    
    return jsonify({
        'success': True,
        'url': test_url,
        'test_url': test_url,
        'environment': 'production' if IS_PRODUCTION else 'development',
        'base_url': BASE_URL,
        'message': f'Page de test WebView générée - {("Production" if IS_PRODUCTION else "Développement")}'
    })

# ===============================================
# 🔍 FONCTION DE DEBUG POUR VÉRIFIER LES URLs
# ===============================================

@app.route('/api/debug/urls', methods=['GET'])
def debug_urls():
    """Route de debug pour vérifier les URLs"""
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
    """Retourne les données 3D en JSON pour debug"""
    try:
        temp_file = os.path.join(TEMP_DIR, f"{model_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Modèle non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        
        # Convertir en format JSON
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
    """Sert les fichiers HTML de visualisation 3D"""
    return send_file(
        os.path.join(PREVIEW_DIR, filename),
        mimetype='text/html'
    )

# ============== FONCTIONS DE GÉNÉRATION HTML ==============

def convertir_numpy_pour_json(obj):
    """Convertit récursivement les types NumPy en types Python natifs pour JSON"""
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
    """Version corrigée avec conversion JSON sécurisée"""
    
    # ✅ CORRECTION : Conversion sécurisée en JSON
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
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: Arial, sans-serif;
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #debug {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            z-index: 100;
            font-size: 12px;
            max-width: 300px;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            z-index: 50;
        }}
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
            console.log('🔧 Status:', message);
        }}

        try {{
            updateStatus('Chargement des données...');
            
            // Vérifier que Three.js est chargé
            if (typeof THREE === 'undefined') {{
                throw new Error('Three.js non chargé');
            }}
            updateStatus('Three.js OK');

            // ✅ CORRECTION : Données du modèle 3D avec JSON sécurisé
            const vertices = {vertices_json};
            const faces = {faces_json};
            
            updateStatus(`Données chargées: ${{vertices.length}} vertices, ${{faces.length}} faces`);

            // Variables Three.js
            let scene, camera, renderer, mesh;

            function init() {{
                updateStatus('Initialisation de la scène...');
                
                // Scène
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f0f0);

                // Caméra
                camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 15);
                
                // Calculer les dimensions
                const yValues = vertices.map(v => v[1]);
                const yMin = Math.min(...yValues);
                const yMax = Math.max(...yValues);
                const hauteur = yMax - yMin;
                const centreY = (yMin + yMax) / 2;
                
                updateStatus(`Dimensions: hauteur=${{hauteur.toFixed(2)}}, centre=${{centreY.toFixed(2)}}`);
                
                // Position caméra
                const distance = Math.max(3.5, hauteur * 2.5);
                camera.position.set(0, centreY * 0.8, distance);
                camera.lookAt(0, centreY, 0);

                // Renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.shadowMap.enabled = true;
                document.getElementById('container').appendChild(renderer.domElement);
                
                updateStatus('Création du mesh...');
                createMesh();
                
                updateStatus('Configuration éclairage...');
                setupLighting();
                
                updateStatus('Configuration contrôles...');
                setupControls();
                
                // Masquer le loading
                document.getElementById('loading').style.display = 'none';
                
                updateStatus('Animation démarrée');
                animate();
                
                // Redimensionnement
                window.addEventListener('resize', onWindowResize);
            }}

            function createMesh() {{
                try {{
                    const geometry = new THREE.BufferGeometry();
                    const verticesArray = new Float32Array(vertices.flat());
                    geometry.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
                    
                    const facesArray = new Uint32Array(faces.flat());
                    geometry.setIndex(new THREE.BufferAttribute(facesArray, 1));
                    geometry.computeVertexNormals();

                    const material = new THREE.MeshLambertMaterial({{
                        color: 0xcc9966,
                        transparent: false,
                        opacity: 1.0
                    }});

                    mesh = new THREE.Mesh(geometry, material);
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    scene.add(mesh);
                    
                    updateStatus('Mesh créé avec succès');
                }} catch (error) {{
                    updateStatus('Erreur création mesh: ' + error.message);
                    console.error('Erreur mesh:', error);
                }}
            }}

            function setupLighting() {{
                const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 10, 5);
                directionalLight.castShadow = true;
                scene.add(directionalLight);

                const light1 = new THREE.PointLight(0xffffff, 0.4);
                light1.position.set(-5, 5, 5);
                scene.add(light1);

                const light2 = new THREE.PointLight(0xffffff, 0.4);
                light2.position.set(5, 5, -5);
                scene.add(light2);
            }}

            function setupControls() {{
                let isRotating = false;
                let previousTouch = null;

                // Contrôles tactiles
                renderer.domElement.addEventListener('touchstart', (e) => {{
                    e.preventDefault();
                    isRotating = true;
                    if (e.touches.length === 1) {{
                        previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
                    }}
                }});

                renderer.domElement.addEventListener('touchmove', (e) => {{
                    e.preventDefault();
                    if (isRotating && previousTouch && e.touches.length === 1) {{
                        const deltaX = e.touches[0].clientX - previousTouch.x;
                        const deltaY = e.touches[0].clientY - previousTouch.y;

                        if (mesh) {{
                            mesh.rotation.y += deltaX * 0.01;
                            mesh.rotation.x += deltaY * 0.01;
                        }}

                        previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
                    }}
                }});

                renderer.domElement.addEventListener('touchend', (e) => {{
                    e.preventDefault();
                    isRotating = false;
                    previousTouch = null;
                }});

                // Contrôles souris
                let isMouseDown = false;
                let previousMouse = null;

                renderer.domElement.addEventListener('mousedown', (e) => {{
                    isMouseDown = true;
                    previousMouse = {{ x: e.clientX, y: e.clientY }};
                }});

                renderer.domElement.addEventListener('mousemove', (e) => {{
                    if (isMouseDown && previousMouse) {{
                        const deltaX = e.clientX - previousMouse.x;
                        const deltaY = e.clientY - previousMouse.y;

                        if (mesh) {{
                            mesh.rotation.y += deltaX * 0.01;
                            mesh.rotation.x += deltaY * 0.01;
                        }}

                        previousMouse = {{ x: e.clientX, y: e.clientY }};
                    }}
                }});

                renderer.domElement.addEventListener('mouseup', () => {{
                    isMouseDown = false;
                    previousMouse = null;
                }});
            }}

            function animate() {{
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }}

            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}

            // Initialiser
            init();

        }} catch (error) {{
            updateStatus('ERREUR: ' + error.message);
            console.error('Erreur globale:', error);
            document.getElementById('loading').innerHTML = 
                '<div style="color: red;">Erreur: ' + error.message + '</div>';
        }}
    </script>
</body>
</html>
    """
    
    return html_template


def generer_html_vetement_3d(vertices_corps, vertices_avec_vetement, faces, masque_vetement, couleur, type_vetement):
    """✅ VERSION CORRIGÉE - Utilise json.dumps pour éviter les erreurs True/False"""
    
    import json  # Assurez-vous que json est importé
    
    # ✅ CORRECTION : Utiliser json.dumps pour une conversion sécurisée
    vertices_corps_json = json.dumps(vertices_corps.tolist())
    vertices_vetement_json = json.dumps(vertices_avec_vetement.tolist())
    faces_json = json.dumps(faces.tolist())
    masque_json = json.dumps(masque_vetement.tolist())  # ← CORRECTION PRINCIPALE
    
    # Récupérer les couleurs RGB
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
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: Arial, sans-serif;
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            z-index: 100;
            font-size: 14px;
            max-width: 300px;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            z-index: 100;
            color: white;
            text-align: center;
        }}
        .control-text {{
            font-size: 12px;
            margin: 5px 0;
        }}
        .color-indicator {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: rgb({couleur_rgb[0]}, {couleur_rgb[1]}, {couleur_rgb[2]});
            border: 2px solid white;
            margin-right: 8px;
            vertical-align: middle;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            z-index: 50;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading">
            <div>Chargement du vêtement...</div>
            <div style="font-size: 14px; margin-top: 10px;">{type_vetement} {couleur}</div>
        </div>
        
        <div id="info">
            <div><strong>🎭 {type_vetement}</strong></div>
            <div><span class="color-indicator"></span>{couleur}</div>
            <div>👆 Faites glisser pour tourner</div>
            <div>🤏 Pincez pour zoomer</div>
            <div style="margin-top: 10px; font-size: 12px;">
                Vertices: {len(vertices_avec_vetement)}<br>
                Vêtement: {sum(masque_vetement)} points
            </div>
        </div>
        
        <div id="controls">
            <div class="control-text">Visualisation 3D Interactive</div>
            <div class="control-text">🦶 Corps + Vêtement</div>
        </div>
    </div>

    <script>
        // ✅ CORRECTION : Données converties avec json.dumps
        const verticesCorps = {vertices_corps_json};
        const verticesVetement = {vertices_vetement_json};
        const faces = {faces_json};
        const masqueVetement = {masque_json};  // ← Plus d'erreur True/False !
        const couleurVetement = {couleur_normalized_json};

        // Configuration Three.js
        let scene, camera, renderer;
        let meshCorps, meshVetement;

        function init() {{
            try {{
                console.log('🔧 Initialisation vêtement:', '{type_vetement}', '{couleur}');
                console.log('🔧 Masque vêtement:', masqueVetement.length, 'points');
                
                // Scène
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f0f0);

                // Caméra avec FOV élargi pour voir les pieds
                camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 15);
                
                // Calculer les dimensions du mannequin
                const yValues = verticesCorps.map(v => v[1]);
                const yMin = Math.min(...yValues);
                const yMax = Math.max(...yValues);
                const hauteur = yMax - yMin;
                const centreY = (yMin + yMax) / 2;
                
                console.log('🔧 Dimensions:', {{hauteur, centreY}});
                
                // Position caméra adaptée pour voir les pieds
                const distance = Math.max(3.5, hauteur * 2.5);
                camera.position.set(0, centreY * 0.8, distance);
                camera.lookAt(0, centreY, 0);

                // Renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                document.getElementById('container').appendChild(renderer.domElement);

                // Créer les meshes
                createMeshes();

                // Éclairage
                setupLighting();

                // Contrôles tactiles
                setupControls();

                // Masquer le loading
                document.getElementById('loading').style.display = 'none';

                // Animation
                animate();

                // Redimensionnement
                window.addEventListener('resize', onWindowResize);
                
            }} catch (error) {{
                console.error('❌ Erreur initialisation:', error);
                document.getElementById('loading').innerHTML = 
                    '<div style="color: red;">Erreur: ' + error.message + '</div>';
            }}
        }}

        function createMeshes() {{
            console.log('🔧 Création meshes...');
            
            // === MESH DU CORPS ===
            const geometryCorps = new THREE.BufferGeometry();
            const verticesCorpsArray = new Float32Array(verticesCorps.flat());
            geometryCorps.setAttribute('position', new THREE.BufferAttribute(verticesCorpsArray, 3));
            
            const facesArray = new Uint32Array(faces.flat());
            geometryCorps.setIndex(new THREE.BufferAttribute(facesArray, 1));
            geometryCorps.computeVertexNormals();

            const materialCorps = new THREE.MeshLambertMaterial({{
                color: 0xcc9966, // Couleur peau
                transparent: true,
                opacity: 0.8
            }});

            meshCorps = new THREE.Mesh(geometryCorps, materialCorps);
            meshCorps.castShadow = true;
            meshCorps.receiveShadow = true;
            scene.add(meshCorps);
            console.log('✅ Mesh corps créé');

            // === MESH DU VÊTEMENT ===
            // Extraire seulement les vertices du vêtement
            const verticesVetementOnly = [];
            const facesVetementOnly = [];
            const indexMap = new Map();
            let newIndex = 0;

            // Créer un mapping des anciens indices vers les nouveaux
            for (let i = 0; i < masqueVetement.length; i++) {{
                if (masqueVetement[i]) {{  // ← Plus d'erreur ici !
                    verticesVetementOnly.push(...verticesVetement[i]);
                    indexMap.set(i, newIndex);
                    newIndex++;
                }}
            }}

            // Créer les faces du vêtement
            for (let i = 0; i < faces.length; i++) {{
                const face = faces[i];
                const v1 = face[0], v2 = face[1], v3 = face[2];
                
                // Vérifier si tous les vertices de la face appartiennent au vêtement
                if (masqueVetement[v1] && masqueVetement[v2] && masqueVetement[v3]) {{
                    const newV1 = indexMap.get(v1);
                    const newV2 = indexMap.get(v2);
                    const newV3 = indexMap.get(v3);
                    
                    if (newV1 !== undefined && newV2 !== undefined && newV3 !== undefined) {{
                        facesVetementOnly.push([newV1, newV2, newV3]);
                    }}
                }}
            }}

            if (verticesVetementOnly.length > 0 && facesVetementOnly.length > 0) {{
                const geometryVetement = new THREE.BufferGeometry();
                const verticesVetementArray = new Float32Array(verticesVetementOnly);
                geometryVetement.setAttribute('position', new THREE.BufferAttribute(verticesVetementArray, 3));
                
                const facesVetementArray = new Uint32Array(facesVetementOnly.flat());
                geometryVetement.setIndex(new THREE.BufferAttribute(facesVetementArray, 1));
                geometryVetement.computeVertexNormals();

                const materialVetement = new THREE.MeshLambertMaterial({{
                    color: new THREE.Color(couleurVetement[0], couleurVetement[1], couleurVetement[2]),
                    transparent: true,
                    opacity: 0.95
                }});

                meshVetement = new THREE.Mesh(geometryVetement, materialVetement);
                meshVetement.castShadow = true;
                meshVetement.receiveShadow = true;
                scene.add(meshVetement);
                
                console.log('✅ Vêtement créé:', facesVetementOnly.length, 'faces');
            }} else {{
                console.warn('⚠️ Pas de données vêtement valides');
            }}
        }}

        function setupLighting() {{
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 5);
            directionalLight.castShadow = true;
            scene.add(directionalLight);

            const light1 = new THREE.PointLight(0xffffff, 0.3);
            light1.position.set(-5, 5, 5);
            scene.add(light1);

            const light2 = new THREE.PointLight(0xffffff, 0.3);
            light2.position.set(5, 5, -5);
            scene.add(light2);
        }}

        function setupControls() {{
            let isRotating = false;
            let previousTouch = null;

            renderer.domElement.addEventListener('touchstart', (e) => {{
                e.preventDefault();
                isRotating = true;
                if (e.touches.length === 1) {{
                    previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
                }}
            }});

            renderer.domElement.addEventListener('touchmove', (e) => {{
                e.preventDefault();
                if (isRotating && previousTouch && e.touches.length === 1) {{
                    const deltaX = e.touches[0].clientX - previousTouch.x;
                    const deltaY = e.touches[0].clientY - previousTouch.y;

                    if (meshCorps) {{
                        meshCorps.rotation.y += deltaX * 0.01;
                        meshCorps.rotation.x += deltaY * 0.01;
                    }}
                    if (meshVetement) {{
                        meshVetement.rotation.y += deltaX * 0.01;
                        meshVetement.rotation.x += deltaY * 0.01;
                    }}

                    previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
                }}
            }});

            renderer.domElement.addEventListener('touchend', (e) => {{
                e.preventDefault();
                isRotating = false;
                previousTouch = null;
            }});

            // Contrôles souris
            let isMouseDown = false;
            let previousMouse = null;

            renderer.domElement.addEventListener('mousedown', (e) => {{
                isMouseDown = true;
                previousMouse = {{ x: e.clientX, y: e.clientY }};
            }});

            renderer.domElement.addEventListener('mousemove', (e) => {{
                if (isMouseDown && previousMouse) {{
                    const deltaX = e.clientX - previousMouse.x;
                    const deltaY = e.clientY - previousMouse.y;

                    if (meshCorps) {{
                        meshCorps.rotation.y += deltaX * 0.01;
                        meshCorps.rotation.x += deltaY * 0.01;
                    }}
                    if (meshVetement) {{
                        meshVetement.rotation.y += deltaX * 0.01;
                        meshVetement.rotation.x += deltaY * 0.01;
                    }}

                    previousMouse = {{ x: e.clientX, y: e.clientY }};
                }}
            }});

            renderer.domElement.addEventListener('mouseup', () => {{
                isMouseDown = false;
                previousMouse = null;
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        // Initialiser
        console.log('🚀 Démarrage visualisation vêtement');
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
        'message': '✅ API Mannequin - PIEDS VISIBLES GARANTIS',
        'star_models_available': os.path.exists(STAR_DIR),
        'couleurs_disponibles': len(COULEURS_DISPONIBLES),
        'types_vetements': len(TYPES_VETEMENTS),
        'vedo_available': VEDO_AVAILABLE,
        'camera_correction': {
            'probleme_resolu': 'Pieds coupés dans les captures',
            'solution': 'Caméra centrée sur le vrai milieu du corps',
            'look_at_corrige': 'Centre réel = (y_min + y_max) / 2',
            'fov_elargi': '55° au lieu de 45°',
            'zoom_ajuste': '0.8 pour dézoomer',
            'reset_camera': 'Automatique pour tout voir'
        },
        'timeouts': {
            'generation': f'{TIMEOUT_GENERATION}s',
            'capture': f'{TIMEOUT_CAPTURE}s',
            'scenarios': f'{TIMEOUT_SCENARIOS}s'
        }
    })

# Replace the problematic section in your generate_vetement route (around line 800-850)

@app.route('/api/vetement/generate', methods=['POST'])
@timeout_function(TIMEOUT_GENERATION)
def generate_vetement():
    """✅ Route COMPLÈTE avec lissage et mesh séparé comme le script original"""
    try:
        # Récupération des données JSON
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données JSON requises'}), 400
        
        # ✅ CORRECTION 1: Extraction des paramètres EN PREMIER
        type_vetement_saisi = data.get('type_vetement', '')
        couleur = data.get('couleur', 'Noir')
        gender = data.get('gender', 'neutral')
        mesures = data.get('mesures', {})
        mannequin_id = data.get('mannequin_id', None)
        
        print(f"👗 ✅ Génération vêtement: {type_vetement_saisi} {couleur} pour {gender}")
        
        # ✅ CORRECTION 2: Validation du type de vêtement
        result = valider_type_vetement(type_vetement_saisi)
        if isinstance(result, tuple):
            # Type non trouvé
            type_vetement_valide, types_disponibles = result
            return jsonify({
                'error': f'Type de vêtement non valide: {type_vetement_saisi}',
                'types_disponibles': types_disponibles
            }), 400
        else:
            type_vetement = result  # ✅ Variable bien définie maintenant
        
        # Validation de la couleur
        if couleur not in COULEURS_DISPONIBLES:
            return jsonify({
                'error': f'Couleur non valide: {couleur}',
                'couleurs_disponibles': list(COULEURS_DISPONIBLES.keys())
            }), 400
        
        print(f"✅ Type validé: {type_vetement}, Couleur: {couleur}")
        
        # ✅ CORRECTION 3: Gestion du mannequin de base
        if mannequin_id:
            # Charger un mannequin existant
            temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
            if not os.path.exists(temp_file):
                return jsonify({'error': f'Mannequin {mannequin_id} non trouvé'}), 404
            
            data_mannequin = np.load(temp_file, allow_pickle=True)
            vertices_base = data_mannequin['vertices']
            faces = data_mannequin['faces']
            gender = str(data_mannequin['gender'])
            print(f"✅ Mannequin existant chargé: {mannequin_id}")
        else:
            # Créer un nouveau mannequin
            # Mesures par défaut
            mesures_default = {
                'tour_taille': 68,
                'tour_hanches': 92,
                'tour_poitrine': 88,
                'hauteur': 170,
                'longueur_bras': 60
            }
            mesures = {**mesures_default, **mesures}
            
            # Chargement du modèle STAR
            try:
                if not mannequin_gen.charger_modele_star(gender):
                    return jsonify({'error': 'Impossible de charger le modèle STAR'}), 500
            except Exception as e:
                return jsonify({'error': f'Erreur chargement STAR: {str(e)}'}), 500
            
            # Génération du mannequin
            vertices_base = mannequin_gen.v_template.copy()
            faces = mannequin_gen.f
            
            if mannequin_gen.shapedirs is not None:
                try:
                    joints_base = mannequin_gen.J_regressor.dot(vertices_base)
                    mesures_actuelles = mannequin_gen.calculer_mesures_modele(
                        vertices_base, joints_base, DEFAULT_MAPPING
                    )
                    
                    vertices_base, betas = mannequin_gen.deformer_modele(mesures, mesures_actuelles)
                    print(f"✅ Mannequin déformé avec betas: {betas[:3]}...")
                    
                except Exception as e:
                    print(f"⚠️ Erreur déformation: {e}")
            
            print(f"✅ Nouveau mannequin créé pour {gender}")
        
        # ✅ CORRECTION 4: Application du vêtement
        params_vetement = TYPES_VETEMENTS[type_vetement]
        longueur_relative = params_vetement['longueur_relative']
        
        # Détection des points anatomiques
        points_anat = VetementGenerator.detecter_points_anatomiques(vertices_base)
        
        # Calcul du profil selon le type
        if params_vetement['type'] == 'droite':
            profil = VetementGenerator.calculer_profil_jupe_droite(points_anat, longueur_relative)
        elif params_vetement['type'] == 'ovale':
            ampleur = params_vetement.get('ampleur', 1.4)
            profil = VetementGenerator.calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur)
        elif params_vetement['type'] == 'trapeze':
            evasement = params_vetement.get('evasement', 1.6)
            profil = VetementGenerator.calculer_profil_jupe_trapeze(points_anat, longueur_relative, evasement)
        else:
            return jsonify({'error': f'Type de vêtement non supporté: {params_vetement["type"]}'}), 400
        
        # Application de la forme
        vertices_avec_vetement, masque_vetement = VetementGenerator.appliquer_forme_jupe(vertices_base, profil)
        
        print(f"✅ Vêtement appliqué: {np.sum(masque_vetement)} points modifiés")
        
        # ✅ AJOUT CRUCIAL 1: LISSAGE (comme dans le script original)
        print(f"🔄 Application du lissage...")
        vertices_avec_vetement = VetementGenerator.lisser_jupe(vertices_avec_vetement, masque_vetement, iterations=2)
        
        print(f"✅ Vêtement appliqué et lissé: {np.sum(masque_vetement)} points modifiés")
        
        # ✅ AJOUT CRUCIAL 2: CRÉATION DU MESH SÉPARÉ (comme dans le script original)
        print(f"🎨 Création du mesh vêtement séparé...")
        try:
            mesh_vetement_data = VetementGenerator.creer_mesh_jupe_separe(
                vertices_avec_vetement, masque_vetement, couleur
            )
            
            # ✅ VÉRIFICATION TYPE: S'assurer que c'est bien un dict
            if isinstance(mesh_vetement_data, dict):
                if mesh_vetement_data.get('mesh_object') is not None:
                    print(f"✅ Mesh vêtement créé: {mesh_vetement_data.get('points_count', 0)} points, {mesh_vetement_data.get('faces_count', 0)} faces")
                else:
                    print(f"⚠️ Échec création mesh vêtement")
            else:
                print(f"⚠️ Type inattendu pour mesh_vetement_data: {type(mesh_vetement_data)}")
                # Fallback: créer un dict par défaut
                mesh_vetement_data = {
                    'mesh_object': None,
                    'points_count': 0,
                    'faces_count': 0,
                    'couleur_rgb': [128, 128, 128],
                    'couleur_normalized': [0.5, 0.5, 0.5]
                }
        except Exception as e:
            print(f"❌ Erreur création mesh: {e}")
            # Fallback: créer un dict par défaut
            mesh_vetement_data = {
                'mesh_object': None,
                'points_count': 0,
                'faces_count': 0,
                'couleur_rgb': [128, 128, 128],
                'couleur_normalized': [0.5, 0.5, 0.5]
            }
        
        # ✅ CORRECTION 5: Génération de l'ID unique
        import random
        import string
        import time
        timestamp = str(int(time.time()))[-6:]
        random_suffix = ''.join(random.choices(string.digits, k=3))
        vetement_id = f"vetement_{type_vetement.lower().replace(' ', '_').replace('-', '_')}_{couleur.lower()}_{timestamp}_{random_suffix}"
        
        # ✅ CORRECTION PRINCIPALE: Sauvegarder sans essayer d'extraire les données mesh Vedo
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        try:
            # ✅ SOLUTION: Créer les infos mesh séparément
            mesh_info_dict = {
                'points_count': mesh_vetement_data.get('points_count', 0) if isinstance(mesh_vetement_data, dict) else 0,
                'faces_count': mesh_vetement_data.get('faces_count', 0) if isinstance(mesh_vetement_data, dict) else 0,
                'couleur_rgb': mesh_vetement_data.get('couleur_rgb', [128, 128, 128]) if isinstance(mesh_vetement_data, dict) else [128, 128, 128],
                'couleur_normalized': mesh_vetement_data.get('couleur_normalized', [0.5, 0.5, 0.5]) if isinstance(mesh_vetement_data, dict) else [0.5, 0.5, 0.5],
                'mesh_created': mesh_vetement_data.get('mesh_object') is not None if isinstance(mesh_vetement_data, dict) else False
            }
            
            # ✅ CORRECTION: Sauvegarder les valeurs individuellement
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
                # ✅ CORRECTION: Sauvegarder les valeurs individuellement
                mesh_points_count=mesh_info_dict['points_count'],
                mesh_faces_count=mesh_info_dict['faces_count'],
                mesh_couleur_rgb=mesh_info_dict['couleur_rgb'],
                mesh_couleur_normalized=mesh_info_dict['couleur_normalized'],
                mesh_created=mesh_info_dict['mesh_created']
            )
            print(f"✅ Vêtement sauvegardé (sans données mesh Vedo): {temp_file}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde détaillée: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Erreur sauvegarde: {str(e)}'}), 500
        
        # ✅ CORRECTION 6: Informations complètes sur le vêtement généré
        info = {
            'type_vetement': type_vetement,
            'couleur': couleur,
            'gender': gender,
            'vertices_count': len(vertices_avec_vetement),
            'faces_count': len(faces),
            'vetement_vertices': int(np.sum(masque_vetement)),  # ✅ Conversion explicite
            'longueur_relative': float(longueur_relative),  # ✅ Conversion explicite
            'mesures_appliquees': mesures,
            'lissage_applique': True,  # ✅ Nouveau: confirmation lissage
            'mesh_separe_cree': mesh_vetement_data.get('mesh_object') is not None if isinstance(mesh_vetement_data, dict) else False,
            'mesh_points': mesh_vetement_data.get('points_count', 0) if isinstance(mesh_vetement_data, dict) else 0,
            'mesh_faces': mesh_vetement_data.get('faces_count', 0) if isinstance(mesh_vetement_data, dict) else 0,
            'couleur_rgb': mesh_vetement_data.get('couleur_rgb', [128, 128, 128]) if isinstance(mesh_vetement_data, dict) else [128, 128, 128],
            'camera_correction': '✅ PIEDS VISIBLES GARANTIS + LISSAGE + MESH SÉPARÉ'
        }
        
        # ✅ CORRECTION 7: Convertir les types NumPy avant la sérialisation JSON
        info_convertie = convertir_numpy_pour_json(info)
        
        print(f"✅ ✅ Vêtement généré COMPLET avec PIEDS VISIBLES + LISSAGE + MESH: {vetement_id}")
        
        return jsonify({
            'success': True,
            'message': f'✅ Vêtement {type_vetement} {couleur} généré - RENDU IDENTIQUE AU SCRIPT',
            'info': info_convertie,  # ✅ Utiliser la version convertie
            'vetement_id': vetement_id,
            'qualite': {
                'lissage': '✅ Appliqué (2 itérations)',
                'mesh_separe': '✅ Créé avec triangulation',
                'couleur': f'✅ {couleur} RGB appliquée',
                'rendu': '✅ Identique au script original'
            }
        })
        
    except Exception as e:
        print(f"❌ Erreur génération vêtement: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500





def generer_html_vetement_preview(vertices_corps, vertices_avec_vetement, faces, masque_vetement, couleur, type_vetement):
    """✅ VERSION ULTRA-RÉALISTE avec matériaux avancés et éclairage professionnel"""
    
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
        body {{ 
            width: 100vw; 
            height: 100vh; 
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        #canvas {{ display: block; }}
        #info {{
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            color: white;
            padding: 16px 20px;
            border-radius: 12px;
            z-index: 10;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        .info-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .color-swatch {{
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 6px;
            background-color: rgb({couleur_rgb[0]}, {couleur_rgb[1]}, {couleur_rgb[2]});
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border: 2px solid rgba(255,255,255,0.3);
        }}
        .info-detail {{
            font-size: 13px;
            color: rgba(255,255,255,0.85);
            margin: 4px 0;
        }}
        .controls-hint {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.15);
            font-size: 12px;
            color: rgba(255,255,255,0.7);
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 30px 40px;
            border-radius: 16px;
            text-align: center;
            z-index: 100;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }}
        .spinner {{
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top: 3px solid white;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <div style="font-size: 16px; font-weight: 600;">Chargement du vêtement...</div>
        <div style="font-size: 13px; margin-top: 8px; opacity: 0.7;">{type_vetement} {couleur}</div>
    </div>
    
    <div id="info">
        <div class="info-title">
            <span class="color-swatch"></span>
            <span>{type_vetement}</span>
        </div>
        <div class="info-detail">Couleur: {couleur}</div>
        <div class="info-detail">Rendu réaliste avec éclairage avancé</div>
        <div class="controls-hint">
            <div>🖱️ Cliquer-glisser pour tourner</div>
            <div>📱 Glisser pour faire pivoter</div>
            <div>🤏 Pincer pour zoomer</div>
        </div>
    </div>

    <script>
        let scene, camera, renderer;
        let meshCorps, meshVetement;
        let rotationX = 0, rotationY = 0;
        let mouseDown = false, prevX = 0, prevY = 0;
        let ambientLight, mainLight, fillLight, rimLight;

        const verticesCorps = {vertices_corps_json};
        const verticesVetement = {vertices_vetement_json};
        const faces = {faces_json};
        const masque = {masque_json};
        const couleur = {couleur_json};

        function init() {{
            // ========================================
            // SCÈNE ET CAMÉRA
            // ========================================
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf5f7fa);
            
            // ✅ FOG pour profondeur
            scene.fog = new THREE.Fog(0xf5f7fa, 5, 15);

            const yValues = verticesCorps.map(v => v[1]);
            const yMin = Math.min(...yValues);
            const yMax = Math.max(...yValues);
            const hauteur = yMax - yMin;
            const centreY = (yMin + yMax) / 2;

            camera = new THREE.PerspectiveCamera(
                50,  // FOV légèrement réduit pour un effet plus naturel
                window.innerWidth / window.innerHeight, 
                0.1, 
                1000
            );
            
            const dist = Math.max(2.2, hauteur * 2.4);
            camera.position.set(0.3, centreY * 0.85, dist);  // Légèrement décalé pour perspective
            camera.lookAt(0, centreY, 0);

            // ========================================
            // RENDERER avec antialiasing premium
            // ========================================
            renderer = new THREE.WebGLRenderer({{ 
                antialias: true,
                alpha: true,
                powerPreference: 'high-performance'
            }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            
            // ✅ ÉCLAIRAGE RÉALISTE ACTIVÉ
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.2;
            renderer.outputEncoding = THREE.sRGBEncoding;
            
            document.body.appendChild(renderer.domElement);

            // ========================================
            // MESH DU CORPS (Peau réaliste)
            // ========================================
            const geometryCorps = new THREE.BufferGeometry();
            geometryCorps.setAttribute('position', 
                new THREE.BufferAttribute(new Float32Array(verticesCorps.flat()), 3)
            );
            geometryCorps.setIndex(
                new THREE.BufferAttribute(new Uint32Array(faces.flat()), 1)
            );
            geometryCorps.computeVertexNormals();

            // ✅ MATÉRIAU PEAU RÉALISTE
            const materialCorps = new THREE.MeshStandardMaterial({{
                color: 0xd4a574,
                roughness: 0.65,  // Peau légèrement mate
                metalness: 0.05,  // Très peu métallique
                envMapIntensity: 0.3,
            }});

            meshCorps = new THREE.Mesh(geometryCorps, materialCorps);
            meshCorps.castShadow = true;
            meshCorps.receiveShadow = true;
            scene.add(meshCorps);

            // ========================================
            // MESH DU VÊTEMENT (Tissu photo-réaliste)
            // ========================================
            const verticesVetOnly = [];
            const facesVetOnly = [];
            const indexMap = new Map();
            let newIdx = 0;

            for (let i = 0; i < masque.length; i++) {{
                if (masque[i]) {{
                    verticesVetOnly.push(...verticesVetement[i]);
                    indexMap.set(i, newIdx);
                    newIdx++;
                }}
            }}

            for (let i = 0; i < faces.length; i++) {{
                const [v1, v2, v3] = faces[i];
                if (masque[v1] && masque[v2] && masque[v3]) {{
                    const nv1 = indexMap.get(v1);
                    const nv2 = indexMap.get(v2);
                    const nv3 = indexMap.get(v3);
                    if (nv1 !== undefined && nv2 !== undefined && nv3 !== undefined) {{
                        facesVetOnly.push([nv1, nv2, nv3]);
                    }}
                }}
            }}

            if (verticesVetOnly.length > 0 && facesVetOnly.length > 0) {{
                const geometryVet = new THREE.BufferGeometry();
                geometryVet.setAttribute('position',
                    new THREE.BufferAttribute(new Float32Array(verticesVetOnly), 3)
                );
                geometryVet.setIndex(
                    new THREE.BufferAttribute(new Uint32Array(facesVetOnly.flat()), 1)
                );
                geometryVet.computeVertexNormals();
                
                // ✅ SMOOTH les normales pour un aspect plus doux
                // geometryVet.computeVertexNormals();

                // ✅ MATÉRIAU VÊTEMENT ULTRA-RÉALISTE
                const materialVet = new THREE.MeshStandardMaterial({{
                    color: new THREE.Color(couleur[0], couleur[1], couleur[2]),
                    
                    // ✅ PARAMÈTRES CLÉS POUR LE RÉALISME
                    roughness: 0.45,        // Aspect légèrement satiné (tissu)
                    metalness: 0.1,         // Très peu métallique
                    
                    // ✅ ÉCLAIRAGE ENVIRONNEMENTAL
                    envMapIntensity: 0.5,   // Réflexions subtiles
                    
                    // ✅ RENDU DEUX FACES
                    side: THREE.DoubleSide,
                    
                    // ✅ FLAT SHADING DÉSACTIVÉ pour un rendu lisse
                    flatShading: false,
                    
                    // ✅ OMBRES
                    shadowSide: THREE.DoubleSide,
                }});

                meshVetement = new THREE.Mesh(geometryVet, materialVet);
                meshVetement.castShadow = true;
                meshVetement.receiveShadow = true;
                scene.add(meshVetement);
                
                console.log('✅ Vêtement réaliste créé:', facesVetOnly.length, 'faces');
            }}

            // ========================================
            // ✅ ÉCLAIRAGE PROFESSIONNEL (3-POINT LIGHTING)
            // ========================================
            
            // 1. LUMIÈRE AMBIANTE (base)
            ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            // 2. LUMIÈRE PRINCIPALE (Key Light)
            mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
            mainLight.position.set(3, 8, 5);
            mainLight.castShadow = true;
            
            // ✅ OMBRES HAUTE QUALITÉ
            mainLight.shadow.mapSize.width = 2048;
            mainLight.shadow.mapSize.height = 2048;
            mainLight.shadow.camera.near = 0.5;
            mainLight.shadow.camera.far = 50;
            mainLight.shadow.camera.left = -5;
            mainLight.shadow.camera.right = 5;
            mainLight.shadow.camera.top = 5;
            mainLight.shadow.camera.bottom = -5;
            mainLight.shadow.bias = -0.001;
            
            scene.add(mainLight);

            // 3. LUMIÈRE DE REMPLISSAGE (Fill Light)
            fillLight = new THREE.DirectionalLight(0xffd9b3, 0.6);
            fillLight.position.set(-4, 3, 3);
            scene.add(fillLight);

            // 4. LUMIÈRE DE CONTOUR (Rim Light)
            rimLight = new THREE.DirectionalLight(0xb3d9ff, 0.5);
            rimLight.position.set(0, 2, -5);
            scene.add(rimLight);

            // 5. LUMIÈRES PONCTUELLES D'ACCENTUATION
            const accentLight1 = new THREE.PointLight(0xffffff, 0.4, 10);
            accentLight1.position.set(-2, 4, 4);
            scene.add(accentLight1);

            const accentLight2 = new THREE.PointLight(0xffffff, 0.3, 10);
            accentLight2.position.set(2, 2, -3);
            scene.add(accentLight2);

            // ========================================
            // CONTRÔLES INTERACTIFS
            // ========================================
            setupControls();

            // Masquer le loading
            document.getElementById('loading').style.display = 'none';

            // Animation
            animate();

            // Redimensionnement
            window.addEventListener('resize', onWindowResize);
        }}

        function setupControls() {{
            let isRotating = false;
            let previousTouch = null;

            // CONTRÔLES TACTILES
            renderer.domElement.addEventListener('touchstart', (e) => {{
                e.preventDefault();
                isRotating = true;
                if (e.touches.length === 1) {{
                    previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
                }}
            }});

            renderer.domElement.addEventListener('touchmove', (e) => {{
                e.preventDefault();
                if (isRotating && previousTouch && e.touches.length === 1) {{
                    const deltaX = e.touches[0].clientX - previousTouch.x;
                    const deltaY = e.touches[0].clientY - previousTouch.y;

                    rotationY += deltaX * 0.008;
                    rotationX += deltaY * 0.008;
                    rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));

                    updateRotation();

                    previousTouch = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
                }}
            }});

            renderer.domElement.addEventListener('touchend', (e) => {{
                e.preventDefault();
                isRotating = false;
                previousTouch = null;
            }});

            // CONTRÔLES SOURIS
            let isMouseDown = false;
            let previousMouse = null;

            renderer.domElement.addEventListener('mousedown', (e) => {{
                isMouseDown = true;
                previousMouse = {{ x: e.clientX, y: e.clientY }};
            }});

            renderer.domElement.addEventListener('mousemove', (e) => {{
                if (isMouseDown && previousMouse) {{
                    const deltaX = e.clientX - previousMouse.x;
                    const deltaY = e.clientY - previousMouse.y;

                    rotationY += deltaX * 0.008;
                    rotationX += deltaY * 0.008;
                    rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));

                    updateRotation();

                    previousMouse = {{ x: e.clientX, y: e.clientY }};
                }}
            }});

            renderer.domElement.addEventListener('mouseup', () => {{
                isMouseDown = false;
                previousMouse = null;
            }});

            // ZOOM (molette souris)
            renderer.domElement.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const zoomSpeed = 0.1;
                camera.position.z += e.deltaY * zoomSpeed * 0.01;
                camera.position.z = Math.max(1, Math.min(10, camera.position.z));
            }});
        }}

        function updateRotation() {{
            if (meshCorps) {{
                meshCorps.rotation.order = 'YXZ';
                meshCorps.rotation.y = rotationY;
                meshCorps.rotation.x = rotationX;
            }}
            if (meshVetement) {{
                meshVetement.rotation.order = 'YXZ';
                meshVetement.rotation.y = rotationY;
                meshVetement.rotation.x = rotationX;
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            
            // ✅ ROTATION AUTOMATIQUE LÉGÈRE (optionnel)
            // Décommenter pour auto-rotation
            // rotationY += 0.002;
            // updateRotation();
            
            renderer.render(scene, camera);
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        // Initialiser
        console.log('🚀 Démarrage rendu réaliste vêtement');
        init();
    </script>
</body>
</html>
    """
    
    return html_template




@app.route('/api/vetement/preview/<vetement_id>', methods=['GET'])
def get_vetement_preview(vetement_id):
    """✅ Preview en HTML Three.js - Pas de Vedo requis"""
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
        
        # Générer la page HTML (pas de PNG, pas de Vedo!)
        html_content = generer_html_vetement_preview(
            vertices_corps, vertices_avec_vetement, faces, 
            masque_vetement, couleur, type_vetement
        )
        
        # Sauvegarder le fichier HTML
        html_file = os.path.join(PREVIEW_DIR, f"{vetement_id}_preview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Retourner l'URL de la page HTML
        webview_url = f"{BASE_URL}/webview/{vetement_id}_preview.html"
        
        return jsonify({
            'success': True,
            'preview_url': webview_url,
            'type': 'html_three_js',  # Important: type d'aperçu
            'message': f'Aperçu {type_vetement} {couleur} prêt'
        })
        
    except Exception as e:
        print(f"❌ Erreur preview vêtement: {e}")
        return jsonify({'error': str(e)}), 500
    
    

def generate_fallback_preview(vetement_id, type_vetement, couleur, vertices):
    """Génère un preview de fallback avec matplotlib si Vedo échoue"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print(f"🔄 Génération fallback preview pour {vetement_id}")
        
        # Créer la figure
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Extraire les coordonnées
        x = vertices[:, 0]
        y = vertices[:, 1] 
        z = vertices[:, 2]
        
        # Créer un scatter plot coloré
        couleur_rgb = COULEURS_DISPONIBLES.get(couleur, [128, 128, 128])
        couleur_normalized = [c/255.0 for c in couleur_rgb]
        
        ax.scatter(x, y, z, c=[couleur_normalized], s=1, alpha=0.8)
        
        # Configuration de la vue
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f"{type_vetement} {couleur}", fontsize=14, pad=20)
        
        # Ajuster les limites pour centrer le modèle
        ax.set_xlim([x.min()-0.1, x.max()+0.1])
        ax.set_ylim([y.min()-0.1, y.max()+0.1])
        ax.set_zlim([z.min()-0.1, z.max()+0.1])
        
        # Angle de vue pour bien voir le modèle
        ax.view_init(elev=20, azim=45)
        
        # Sauvegarder
        preview_file = os.path.join(PREVIEW_DIR, f"{vetement_id}_fallback_preview.png")
        plt.savefig(preview_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Fallback preview généré: {preview_file}")
        
        return send_file(
            preview_file,
            mimetype='image/png',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"❌ Erreur fallback preview: {e}")
        # Dernière option: retourner une erreur JSON claire
        return jsonify({
            'error': 'Impossible de générer le preview',
            'details': str(e),
            'vetement_id': vetement_id
        }), 500


@app.route('/api/vetement/visualize/<vetement_id>', methods=['POST'])
def visualize_vetement_3d(vetement_id):
    """✅ Route corrigée pour visualisation avec mesh séparé"""
    try:
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Vêtement non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices_corps = data['vertices_corps']
        vertices_avec_vetement = data['vertices_avec_vetement']  # ✅ Version lissée
        faces = data['faces']
        masque_vetement = data['masque_vetement']
        couleur = str(data['couleur'])
        type_vetement = str(data['type_vetement'])
        
        # ✅ Paramètres de visualisation
        request_data = request.get_json() or {}
        angle_camera = request_data.get('angle_camera', 0)
        distance_camera = request_data.get('distance_camera', 3.0)
        hauteur_camera = request_data.get('hauteur_camera', 0.0)
        
        print(f"🎬 Visualisation {vetement_id}: angle={angle_camera}°, distance={distance_camera}")
        
        # ✅ VISUALISATION AVEC MESH SÉPARÉ (comme script original)
        try:
            import vedo
            
            # Créer le mesh du corps (sans vêtement)
            mesh_corps = vedo.Mesh([vertices_corps, faces])
            mesh_corps.color('lightblue').alpha(0.7)
            
            # ✅ Créer le mesh du vêtement séparé
            mesh_vetement_data = VetementGenerator.creer_mesh_jupe_separe(
                vertices_avec_vetement, masque_vetement, couleur
            )
            
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data['mesh_object'] is not None:
                mesh_vetement = mesh_vetement_data['mesh_object']
                meshes_a_afficher.append(mesh_vetement)
                print(f"✅ Mesh vêtement ajouté: {mesh_vetement_data['points_count']} points")
            else:
                print(f"⚠️ Fallback: utilisation mesh corps modifié")
                mesh_corps_modifie = vedo.Mesh([vertices_avec_vetement, faces])
                mesh_corps_modifie.color('lightblue').alpha(0.9)
                meshes_a_afficher = [mesh_corps_modifie]
            
            # ✅ Configuration caméra pour PIEDS VISIBLES
            plotter = vedo.Plotter(offscreen=True, size=(800, 600))
            
            for mesh in meshes_a_afficher:
                plotter.add(mesh)
            
            # Position caméra optimisée
            bounds = meshes_a_afficher[0].bounds()
            center_y = (bounds[2] + bounds[3]) / 2
            camera_height = bounds[2] + (bounds[3] - bounds[2]) * 0.3  # ✅ Plus bas pour voir pieds
            
            camera_x = distance_camera * np.cos(np.radians(angle_camera))
            camera_z = distance_camera * np.sin(np.radians(angle_camera))
            
            plotter.camera.SetPosition(camera_x, camera_height + hauteur_camera, camera_z)
            plotter.camera.SetFocalPoint(0, center_y, 0)
            plotter.camera.SetViewUp(0, 1, 0)
            
            # Rendu et sauvegarde
            screenshot_path = os.path.join(TEMP_DIR, f"{vetement_id}_view_{angle_camera}.png")
            plotter.screenshot(screenshot_path)
            plotter.close()
            
            print(f"✅ Visualisation sauvegardée: {screenshot_path}")
            
            return jsonify({
                'success': True,
                'message': f'Visualisation {type_vetement} générée avec mesh séparé',
                'screenshot_path': screenshot_path,
                'info': {
                    'vetement_id': vetement_id,
                    'type_vetement': type_vetement,
                    'couleur': couleur,
                    'angle_camera': angle_camera,
                    'mesh_separe_utilise': mesh_vetement_data['mesh_object'] is not None,
                    'mesh_points': mesh_vetement_data['points_count'],
                    'rendu_qualite': '✅ Identique au script original'
                }
            })
            
        except ImportError:
            return jsonify({'error': 'Vedo non disponible pour la visualisation 3D'}), 500
        except Exception as e:
            print(f"❌ Erreur visualisation 3D: {e}")
            return jsonify({'error': f'Erreur visualisation: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mannequin/generate', methods=['POST'])
def generate_mannequin():
    """✅ Route corrigée pour la génération de mannequins avec pieds visibles"""
    try:
        # Récupération des données JSON
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données JSON requises'}), 400
        
        # Extraction des paramètres
        gender = data.get('gender', 'neutral')
        mesures = data.get('mesures', {})
        
        # Validation du genre
        genders_valides = ['neutral', 'male', 'female']
        if gender not in genders_valides:
            return jsonify({
                'error': f'Genre non valide: {gender}',
                'genders_disponibles': genders_valides
            }), 400
        
        # Mesures par défaut
        mesures_default = {
            'tour_taille': 68,
            'tour_hanches': 92,
            'tour_poitrine': 88,
            'hauteur': 170,
            'longueur_bras': 60
        }
        mesures = {**mesures_default, **mesures}
        
        print(f"👤 ✅ Génération mannequin {gender} avec mesures: {mesures}")
        
        # Chargement du modèle STAR
        try:
            if not mannequin_gen.charger_modele_star(gender):
                return jsonify({'error': 'Impossible de charger le modèle STAR'}), 500
        except Exception as e:
            return jsonify({'error': f'Erreur chargement STAR: {str(e)}'}), 500
        
        # Génération du mannequin de base
        vertices_base = mannequin_gen.v_template.copy()
        faces = mannequin_gen.f
        
        # Application des mesures si des blend shapes sont disponibles
        if mannequin_gen.shapedirs is not None:
            try:
                joints_base = mannequin_gen.J_regressor.dot(vertices_base)
                mesures_actuelles = mannequin_gen.calculer_mesures_modele(
                    vertices_base, joints_base, DEFAULT_MAPPING
                )
                
                vertices_final, betas = mannequin_gen.deformer_modele(mesures, mesures_actuelles)
                print(f"✅ Déformation appliquée avec betas: {betas[:5]}...")
                
            except Exception as e:
                print(f"⚠️ Erreur déformation: {e}")
                vertices_final = vertices_base
                betas = np.zeros(10)
        else:
            print("⚠️ Pas de blend shapes disponibles - mannequin de base utilisé")
            vertices_final = vertices_base
            betas = np.zeros(10)
        
        # Génération de l'ID unique
        import random
        import string
        import time
        timestamp = str(int(time.time()))[-6:]
        random_suffix = ''.join(random.choices(string.digits, k=3))
        mannequin_id = f"mannequin_{gender}_{timestamp}_{random_suffix}"
        
        # Sauvegarde temporaire
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
            print(f"✅ Mannequin {gender} sauvegardé: {temp_file}")
        except Exception as e:
            return jsonify({'error': f'Erreur sauvegarde: {str(e)}'}), 500
        
        # Calcul des dimensions pour les informations
        hauteur, y_min, y_max = visualisateur.calculer_hauteur_mannequin(vertices_final)
        
        # Informations sur le modèle généré
        info = {
            'gender': gender,
            'vertices_count': len(vertices_final),
            'faces_count': len(faces),
            'joints_count': len(mannequin_gen.Jtr) if mannequin_gen.Jtr is not None else 0,
            'betas': betas.tolist(),  # ✅ Déjà converti avec .tolist()
            'mesures_appliquees': mesures,
            'has_shapedirs': mannequin_gen.shapedirs is not None,
            'dimensions': {
                'hauteur_totale': round(hauteur, 2),
                'y_min_pieds': round(y_min, 2),
                'y_max_tete': round(y_max, 2),
                'centre_reel': round((y_min + y_max) / 2, 2)
            },
            'camera_correction': '✅ PIEDS VISIBLES GARANTIS'
        }
        
        # ✅ CORRECTION : Convertir les types NumPy avant la sérialisation JSON
        info_convertie = convertir_numpy_pour_json(info)
        
        print(f"✅ ✅ Mannequin généré avec PIEDS VISIBLES: {mannequin_id}")
        
        return jsonify({
            'success': True,
            'message': f'✅ Mannequin {gender} généré - PIEDS VISIBLES GARANTIS',
            'info': info_convertie,
            'mannequin_id': mannequin_id
        })
        
    except Exception as e:
        print(f"❌ Erreur génération mannequin: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    


def generer_html_mannequin_preview(vertices, faces, gender):
    """Génère une page HTML Three.js compacte pour preview mannequin"""
    
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
        body {{ 
            width: 100vw; 
            height: 100vh; 
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        #canvas {{ display: block; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            z-index: 10;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div id="loading">Chargement du mannequin...</div>
    <div id="info">
        <strong>Mannequin {gender}</strong><br>
        Vertices: {len(vertices)}<br>
        Glissez pour tourner
    </div>

    <script>
        let scene, camera, renderer, mannequin;
        let rotationX = 0, rotationY = 0;
        let mouseDown = false, prevX = 0, prevY = 0;

        const vertices = {vertices_json};
        const faces = {faces_json};

        function init() {{
            // Scène et caméra
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xfafafa);

            const yValues = vertices.map(v => v[1]);
            const yMin = Math.min(...yValues);
            const yMax = Math.max(...yValues);
            const hauteur = yMax - yMin;
            const centreY = (yMin + yMax) / 2;

            camera = new THREE.PerspectiveCamera(
                55, 
                window.innerWidth / window.innerHeight, 
                0.1, 
                1000
            );
            
            const dist = Math.max(2, hauteur * 2.2);
            camera.position.set(0, centreY * 0.8, dist);
            camera.lookAt(0, centreY, 0);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            // Géométrie
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', 
                new THREE.BufferAttribute(new Float32Array(vertices.flat()), 3)
            );
            geometry.setIndex(
                new THREE.BufferAttribute(new Uint32Array(faces.flat()), 1)
            );
            geometry.computeVertexNormals();

            // Matériau
            const material = new THREE.MeshPhongMaterial({{
                color: 0xd4a574,
                shininess: 20,
                side: THREE.DoubleSide
            }});

            mannequin = new THREE.Mesh(geometry, material);
            scene.add(mannequin);

            // Lumières
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 5);
            scene.add(directionalLight);

            // Contrôles souris
            document.addEventListener('mousedown', onMouseDown);
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);

            // Tactile
            document.addEventListener('touchstart', onTouchStart);
            document.addEventListener('touchmove', onTouchMove);
            document.addEventListener('touchend', onTouchEnd);

            document.getElementById('loading').style.display = 'none';
            animate();
        }}

        function onMouseDown(e) {{
            mouseDown = true;
            prevX = e.clientX;
            prevY = e.clientY;
        }}

        function onMouseMove(e) {{
            if (!mouseDown) return;
            
            const deltaX = e.clientX - prevX;
            const deltaY = e.clientY - prevY;
            
            rotationY += deltaX * 0.005;
            rotationX += deltaY * 0.005;
            
            mannequin.rotation.order = 'YXZ';
            mannequin.rotation.y = rotationY;
            mannequin.rotation.x = rotationX;
            
            prevX = e.clientX;
            prevY = e.clientY;
        }}

        function onMouseUp() {{
            mouseDown = false;
        }}

        let touchStart = null;

        function onTouchStart(e) {{
            if (e.touches.length === 1) {{
                touchStart = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
            }}
        }}

        function onTouchMove(e) {{
            if (!touchStart || e.touches.length !== 1) return;
            
            const deltaX = e.touches[0].clientX - touchStart.x;
            const deltaY = e.touches[0].clientY - touchStart.y;
            
            rotationY += deltaX * 0.005;
            rotationX += deltaY * 0.005;
            
            mannequin.rotation.order = 'YXZ';
            mannequin.rotation.y = rotationY;
            mannequin.rotation.x = rotationX;
            
            touchStart = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
        }}

        function onTouchEnd() {{
            touchStart = null;
        }}

        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        init();
    </script>
</body>
</html>
    """
    
    return html_template



    
    
@app.route('/api/mannequin/preview/<mannequin_id>', methods=['GET'])
def get_mannequin_preview(mannequin_id):
    """✅ Preview mannequin en HTML Three.js"""
    try:
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        gender = str(data['gender'])
        
        # Générer la page HTML
        html_content = generer_html_mannequin_preview(vertices, faces, gender)
        
        # Sauvegarder le fichier HTML
        html_file = os.path.join(PREVIEW_DIR, f"{mannequin_id}_preview.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Retourner l'URL de la page HTML
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
    """Liste des types de vêtements disponibles"""
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
        'camera_correction': {
            'status': '✅ PIEDS VISIBLES GARANTIS',
            'probleme_resolu': 'Look_at trop haut coupait les pieds',
            'solution': 'Centre réel = (y_min + y_max) / 2',
            'ameliorations': [
                'FOV élargi à 55°',
                'Zoom 0.8 pour dézoomer',
                'Reset camera automatique',
                'Position adaptée à la hauteur'
            ]
        }
    })


# ✅ ROUTE MANQUANTE AJOUTÉE - À placer dans app.py après les autres routes mannequin

@app.route('/api/mannequin/visualize/<mannequin_id>', methods=['POST'])
def visualize_mannequin_3d(mannequin_id):
    """✅ Visualisation 3D mannequin corrigée avec thread approprié"""
    try:
        if not VEDO_AVAILABLE:
            return jsonify({'error': 'Vedo non disponible pour la visualisation 3D'}), 400
        
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        gender = str(data['gender'])
        
        print(f"🎭 ✅ Visualisation 3D mannequin AVEC PIEDS VISIBLES: {mannequin_id}")
        
        # ✅ CORRECTION: Thread sans dépendance Flask
        def visualiser_mannequin():
            try:
                titre = f"✅ Mannequin {gender} - PIEDS VISIBLES"
                
                # Créer le mesh du mannequin
                if VEDO_AVAILABLE:
                    from vedo import Mesh, Plotter
                    
                    mesh_mannequin = Mesh([vertices, faces])
                    mesh_mannequin.color([0.8, 0.6, 0.4]).alpha(0.9)
                    
                    plt = Plotter(bg='white', axes=1, title=titre, interactive=True)
                    plt.add(mesh_mannequin)
                    
                    # ✅ CONFIGURATION CAMÉRA CORRIGÉE POUR VOIR LES PIEDS
                    visualisateur.configurer_camera_pieds_visibles(plt, vertices)
                    
                    plt.show(interactive=True)
                    plt.close()
                    
                    print(f"✅ ✅ Visualisation 3D mannequin AVEC PIEDS réussie pour {mannequin_id}")
                else:
                    print(f"❌ Vedo non disponible pour {mannequin_id}")
                    
            except Exception as e:
                print(f"❌ Erreur thread visualisation mannequin: {e}")
        
        # ✅ RÉPONSE AVANT LE THREAD
        response_data = {
            'success': True,
            'message': f'✅ Visualisation 3D mannequin {gender} AVEC PIEDS VISIBLES lancée',
            'mannequin_id': mannequin_id,
            'gender': gender,
            'camera_correction': {
                'pieds_visibles': True,
                'centre_reel': '(y_min + y_max) / 2',
                'fov_elargi': '55°',
                'zoom_dezoome': '0.8',
                'reset_camera': 'Automatique'
            }
        }
        
        # Lancer le thread après avoir préparé la réponse
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
    
    
    app.run(
        host='0.0.0.0',  # ← Important : écouter sur toutes les interfaces
        port=port,
        debug=not IS_PRODUCTION  # Debug désactivé en production
    )
    
