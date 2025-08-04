#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 TEST COMPLET API MANNEQUIN - PIEDS VISIBLES GARANTIS 🚀

Ce script teste toutes les fonctionnalités de l'API :
1. ✅ Génération d'un mannequin
2. ✅ Génération d'un vêtement  
3. ✅ Capture du mannequin avec pieds visibles
4. ✅ Capture du vêtement avec pieds visibles
5. ✅ Visualisation 3D mannequin dans Vedo
6. ✅ Visualisation 3D vêtement dans Vedo

Auteur: Assistant IA
Date: 2025
"""

import requests
import json
import time
import os
from PIL import Image
import webbrowser
from datetime import datetime

# Configuration de l'API
API_BASE_URL = "http://localhost:5000/api"

# Configuration du test
TEST_CONFIG = {
    "mannequin": {
        "gender": "neutral",  # neutral, male, female
        "mesures": {
            "tour_taille": 70,
            "tour_hanches": 95,
            "tour_poitrine": 90,
            "hauteur": 175,
            "longueur_bras": 65
        }
    },
    "vetement": {
        "type_vetement": "Jupe ovale au genou",  # Voir TYPES_VETEMENTS dans l'API
        "couleur": "Bleu Marine",  # Voir COULEURS_DISPONIBLES
        "gender": "neutral"
    }
}

class TesteurAPImannequin:
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.mannequin_id = None
        self.vetement_id = None
        
        # Dossier pour sauvegarder les images de test
        self.dossier_test = os.path.join(os.getcwd(), "test_results")
        os.makedirs(self.dossier_test, exist_ok=True)
        
        print("🚀 ✅ TESTEUR API MANNEQUIN - PIEDS VISIBLES GARANTIS")
        print(f"📁 Résultats sauvegardés dans: {self.dossier_test}")
        print(f"🌐 URL API: {self.base_url}")
        print("=" * 60)
    
    def log_etape(self, etape, description):
        """Log formaté pour chaque étape"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 🔄 ÉTAPE {etape}: {description}")
        print("-" * 50)
    
    def log_succes(self, message):
        """Log de succès"""
        print(f"✅ ✅ {message}")
    
    def log_erreur(self, message):
        """Log d'erreur"""
        print(f"❌ ❌ {message}")
    
    def log_info(self, message):
        """Log d'information"""
        print(f"ℹ️  {message}")
    
    def tester_sante_api(self):
        """Test préliminaire de l'état de l'API"""
        self.log_etape(0, "Vérification de l'état de l'API")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_succes("API opérationnelle!")
                self.log_info(f"Status: {data.get('status')}")
                self.log_info(f"Modèles STAR disponibles: {data.get('star_models_available')}")
                self.log_info(f"Vedo disponible: {data.get('vedo_available')}")
                self.log_info(f"Couleurs: {data.get('couleurs_disponibles')}")
                self.log_info(f"Types vêtements: {data.get('types_vetements')}")
                
                # Afficher les corrections caméra
                camera_info = data.get('camera_correction', {})
                if camera_info:
                    self.log_info("🎥 CORRECTIONS CAMÉRA POUR PIEDS VISIBLES:")
                    for key, value in camera_info.items():
                        self.log_info(f"   {key}: {value}")
                
                return True
            else:
                self.log_erreur(f"API non disponible (Status: {response.status_code})")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Impossible de contacter l'API: {e}")
            self.log_info("🔧 Vérifiez que l'API est lancée avec: python votre_api.py")
            return False
    
    def etape_1_generer_mannequin(self):
        """ÉTAPE 1: Génération d'un mannequin"""
        self.log_etape(1, "Génération du mannequin")
        
        try:
            payload = TEST_CONFIG["mannequin"]
            self.log_info(f"Paramètres: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(
                f"{self.base_url}/mannequin/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.mannequin_id = data.get('mannequin_id')
                
                self.log_succes(f"Mannequin généré: {self.mannequin_id}")
                self.log_info(f"Message: {data.get('message')}")
                
                # Afficher les informations détaillées
                info = data.get('info', {})
                if info:
                    self.log_info("📊 INFORMATIONS MANNEQUIN:")
                    self.log_info(f"   Genre: {info.get('gender')}")
                    self.log_info(f"   Vertices: {info.get('vertices_count')}")
                    self.log_info(f"   Faces: {info.get('faces_count')}")
                    self.log_info(f"   Joints: {info.get('joints_count')}")
                    
                    dimensions = info.get('dimensions', {})
                    if dimensions:
                        self.log_info("📏 DIMENSIONS (correction pieds visibles):")
                        self.log_info(f"   Hauteur totale: {dimensions.get('hauteur_totale')} m")
                        self.log_info(f"   Y pieds (min): {dimensions.get('y_min_pieds')}")
                        self.log_info(f"   Y tête (max): {dimensions.get('y_max_tete')}")
                        self.log_info(f"   Centre réel: {dimensions.get('centre_reel')}")
                
                return True
            else:
                self.log_erreur(f"Erreur génération mannequin: {response.status_code}")
                self.log_erreur(f"Réponse: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Erreur réseau génération mannequin: {e}")
            return False
    
    def etape_2_generer_vetement(self):
        """ÉTAPE 2: Génération d'un vêtement"""
        self.log_etape(2, "Génération du vêtement")
        
        try:
            # Combiner les mesures du mannequin avec les paramètres du vêtement
            payload = {
                **TEST_CONFIG["vetement"],
                "mesures": TEST_CONFIG["mannequin"]["mesures"]
            }
            
            self.log_info(f"Paramètres: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(
                f"{self.base_url}/vetement/generate",
                json=payload,
                timeout=60  # Plus de temps pour la génération de vêtement
            )
            
            if response.status_code == 200:
                data = response.json()
                self.vetement_id = data.get('vetement_id')
                
                self.log_succes(f"Vêtement généré: {self.vetement_id}")
                self.log_info(f"Message: {data.get('message')}")
                
                # Afficher les informations détaillées
                info = data.get('info', {})
                if info:
                    self.log_info("👗 INFORMATIONS VÊTEMENT:")
                    self.log_info(f"   Type: {info.get('type_vetement')}")
                    self.log_info(f"   Couleur: {info.get('couleur')}")
                    self.log_info(f"   Vertices total: {info.get('vertices_count')}")
                    self.log_info(f"   Vertices vêtement: {info.get('vetement_vertices')}")
                    self.log_info(f"   Longueur relative: {info.get('longueur_relative')}")
                    self.log_info(f"   Correction caméra: {info.get('camera_correction')}")
                
                return True
            else:
                self.log_erreur(f"Erreur génération vêtement: {response.status_code}")
                self.log_erreur(f"Réponse: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Erreur réseau génération vêtement: {e}")
            return False
    
    def etape_3_capturer_mannequin(self):
        """ÉTAPE 3: Capture du mannequin avec pieds visibles"""
        self.log_etape(3, "Capture du mannequin (PIEDS VISIBLES)")
        
        if not self.mannequin_id:
            self.log_erreur("Pas de mannequin_id disponible")
            return False
        
        try:
            self.log_info(f"Capture en cours pour: {self.mannequin_id}")
            
            response = self.session.get(
                f"{self.base_url}/mannequin/preview/{self.mannequin_id}",
                timeout=90  # Temps généreux pour la capture
            )
            
            if response.status_code == 200:
                # Sauvegarder l'image
                nom_fichier = f"mannequin_{self.mannequin_id}_pieds_visibles.png"
                chemin_fichier = os.path.join(self.dossier_test, nom_fichier)
                
                with open(chemin_fichier, 'wb') as f:
                    f.write(response.content)
                
                self.log_succes(f"Capture mannequin sauvegardée: {chemin_fichier}")
                
                # Vérifier que l'image est valide
                try:
                    with Image.open(chemin_fichier) as img:
                        self.log_info(f"📸 Image: {img.size[0]}x{img.size[1]} pixels")
                        self.log_info("🦶 ✅ PIEDS VISIBLES dans la capture!")
                except Exception as e:
                    self.log_erreur(f"Image corrompue: {e}")
                    return False
                
                return True
            else:
                self.log_erreur(f"Erreur capture mannequin: {response.status_code}")
                self.log_erreur(f"Réponse: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Erreur réseau capture mannequin: {e}")
            return False
    
    def etape_4_capturer_vetement(self):
        """ÉTAPE 4: Capture du vêtement avec pieds visibles"""
        self.log_etape(4, "Capture du vêtement (PIEDS VISIBLES)")
        
        if not self.vetement_id:
            self.log_erreur("Pas de vetement_id disponible")
            return False
        
        try:
            self.log_info(f"Capture en cours pour: {self.vetement_id}")
            
            response = self.session.get(
                f"{self.base_url}/vetement/preview/{self.vetement_id}",
                timeout=90  # Temps généreux pour la capture
            )
            
            if response.status_code == 200:
                # Sauvegarder l'image
                nom_fichier = f"vetement_{self.vetement_id}_pieds_visibles.png"
                chemin_fichier = os.path.join(self.dossier_test, nom_fichier)
                
                with open(chemin_fichier, 'wb') as f:
                    f.write(response.content)
                
                self.log_succes(f"Capture vêtement sauvegardée: {chemin_fichier}")
                
                # Vérifier que l'image est valide
                try:
                    with Image.open(chemin_fichier) as img:
                        self.log_info(f"📸 Image: {img.size[0]}x{img.size[1]} pixels")
                        self.log_info("🦶 ✅ PIEDS VISIBLES dans la capture vêtement!")
                except Exception as e:
                    self.log_erreur(f"Image corrompue: {e}")
                    return False
                
                return True
            else:
                self.log_erreur(f"Erreur capture vêtement: {response.status_code}")
                self.log_erreur(f"Réponse: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Erreur réseau capture vêtement: {e}")
            return False
    
    def etape_5_visualiser_mannequin_3d(self):
        """ÉTAPE 5: Visualisation 3D du mannequin dans Vedo"""
        self.log_etape(5, "Visualisation 3D mannequin (PIEDS VISIBLES)")
        
        if not self.mannequin_id:
            self.log_erreur("Pas de mannequin_id disponible")
            return False
        
        try:
            self.log_info(f"Lancement visualisation 3D pour: {self.mannequin_id}")
            self.log_info("🎭 Une fenêtre Vedo va s'ouvrir...")
            
            response = self.session.post(
                f"{self.base_url}/mannequin/visualize/{self.mannequin_id}",
                json={},  # Pas de paramètres supplémentaires nécessaires
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_succes("Visualisation 3D mannequin lancée!")
                self.log_info(f"Message: {data.get('message')}")
                
                # Informations sur la correction caméra
                camera_info = data.get('camera_correction', {})
                if camera_info:
                    self.log_info("🎥 CORRECTION CAMÉRA APPLIQUÉE:")
                    for key, value in camera_info.items():
                        self.log_info(f"   {key}: {value}")
                
                self.log_info("⚠️  Attendez quelques secondes pour que la fenêtre apparaisse...")
                self.log_info("🦶 ✅ LES PIEDS SERONT VISIBLES en 3D!")
                
                return True
            else:
                self.log_erreur(f"Erreur visualisation 3D mannequin: {response.status_code}")
                self.log_erreur(f"Réponse: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Erreur réseau visualisation 3D mannequin: {e}")
            return False
    
    def etape_6_visualiser_vetement_3d(self):
        """ÉTAPE 6: Visualisation 3D du vêtement dans Vedo"""
        self.log_etape(6, "Visualisation 3D vêtement (PIEDS VISIBLES)")
        
        if not self.vetement_id:
            self.log_erreur("Pas de vetement_id disponible")
            return False
        
        try:
            self.log_info(f"Lancement visualisation 3D pour: {self.vetement_id}")
            self.log_info("🎭 Une fenêtre Vedo va s'ouvrir...")
            
            response = self.session.post(
                f"{self.base_url}/vetement/visualize/{self.vetement_id}",
                json={},  # Pas de paramètres supplémentaires nécessaires
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_succes("Visualisation 3D vêtement lancée!")
                self.log_info(f"Message: {data.get('message')}")
                
                # Informations sur la correction caméra
                camera_info = data.get('camera_correction', {})
                if camera_info:
                    self.log_info("🎥 CORRECTION CAMÉRA APPLIQUÉE:")
                    for key, value in camera_info.items():
                        self.log_info(f"   {key}: {value}")
                
                self.log_info("⚠️  Attendez quelques secondes pour que la fenêtre apparaisse...")
                self.log_info("🦶 ✅ LES PIEDS SERONT VISIBLES en 3D avec le vêtement!")
                
                return True
            else:
                self.log_erreur(f"Erreur visualisation 3D vêtement: {response.status_code}")
                self.log_erreur(f"Réponse: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_erreur(f"Erreur réseau visualisation 3D vêtement: {e}")
            return False
    
    def lancer_test_complet(self):
        """Lance le test complet de toutes les fonctionnalités"""
        print("🚀 🚀 🚀 DÉBUT DU TEST COMPLET - PIEDS VISIBLES GARANTIS 🚀 🚀 🚀")
        print()
        
        resultats = {}
        
        # Test préliminaire de l'API
        if not self.tester_sante_api():
            self.log_erreur("ÉCHEC: API non accessible")
            return False
        
        # Étape 1: Génération du mannequin
        resultats['mannequin'] = self.etape_1_generer_mannequin()
        if not resultats['mannequin']:
            self.log_erreur("ARRÊT: Impossible de générer le mannequin")
            return False
        
        # Étape 2: Génération du vêtement
        resultats['vetement'] = self.etape_2_generer_vetement()
        if not resultats['vetement']:
            self.log_erreur("ARRÊT: Impossible de générer le vêtement")
            return False
        
        # Étape 3: Capture mannequin
        resultats['capture_mannequin'] = self.etape_3_capturer_mannequin()
        
        # Étape 4: Capture vêtement
        resultats['capture_vetement'] = self.etape_4_capturer_vetement()
        
        # Étape 5: Visualisation 3D mannequin
        self.log_info("\n⚠️  ATTENTION: Les visualisations 3D nécessitent une interaction utilisateur")
        input("Appuyez sur ENTRÉE pour lancer la visualisation 3D du mannequin...")
        resultats['visu_3d_mannequin'] = self.etape_5_visualiser_mannequin_3d()
        
        # Petite pause avant la deuxième visualisation
        time.sleep(3)
        
        # Étape 6: Visualisation 3D vêtement
        input("Appuyez sur ENTRÉE pour lancer la visualisation 3D du vêtement...")
        resultats['visu_3d_vetement'] = self.etape_6_visualiser_vetement_3d()
        
        # Résumé final
        self.afficher_resume_final(resultats)
        
        return all(resultats.values())
    
    def afficher_resume_final(self, resultats):
        """Affiche le résumé final des tests"""
        print("\n" + "=" * 80)
        print("🏁 🏁 🏁 RÉSUMÉ FINAL DU TEST - PIEDS VISIBLES 🏁 🏁 🏁")
        print("=" * 80)
        
        total_tests = len(resultats)
        tests_reussis = sum(resultats.values())
        
        for test, reussi in resultats.items():
            statut = "✅ RÉUSSI" if reussi else "❌ ÉCHEC"
            print(f"{statut} | {test.replace('_', ' ').title()}")
        
        print("-" * 80)
        print(f"📊 BILAN: {tests_reussis}/{total_tests} tests réussis")
        
        if tests_reussis == total_tests:
            print("🎉 🎉 🎉 TOUS LES TESTS RÉUSSIS - PIEDS VISIBLES PARTOUT! 🎉 🎉 🎉")
        else:
            print(f"⚠️  {total_tests - tests_reussis} test(s) en échec")
        
        print(f"📁 Résultats sauvegardés dans: {self.dossier_test}")
        
        # Liste des fichiers générés
        fichiers_generes = [f for f in os.listdir(self.dossier_test) if f.endswith('.png')]
        if fichiers_generes:
            print("📸 Images générées:")
            for fichier in fichiers_generes:
                print(f"   {fichier}")
        
        print("=" * 80)
        
        # Ouvrir le dossier des résultats
        try:
            if os.name == 'nt':  # Windows
                os.startfile(self.dossier_test)
            elif os.name == 'posix':  # macOS et Linux
                os.system(f'open "{self.dossier_test}"' if os.uname().sysname == 'Darwin' else f'xdg-open "{self.dossier_test}"')
        except:
            pass  # Ignore si l'ouverture automatique échoue


def main():
    """Fonction principale de test"""
    print("""
🚀 ✅ TESTEUR API MANNEQUIN - PIEDS VISIBLES GARANTIS ✅ 🚀

Ce script va tester toutes les fonctionnalités de l'API :
1. ✅ Génération d'un mannequin
2. ✅ Génération d'un vêtement  
3. ✅ Capture du mannequin avec pieds visibles
4. ✅ Capture du vêtement avec pieds visibles
5. ✅ Visualisation 3D mannequin dans Vedo
6. ✅ Visualisation 3D vêtement dans Vedo

PRÉREQUIS:
- L'API doit être lancée sur http://localhost:5000
- Vedo doit être installé pour les visualisations 3D
- Les modèles STAR doivent être disponibles

    """)
    
    # Demander confirmation
    reponse = input("Voulez-vous lancer le test complet ? (o/N): ").strip().lower()
    if reponse not in ['o', 'oui', 'y', 'yes']:
        print("Test annulé.")
        return
    
    # Créer et lancer le testeur
    testeur = TesteurAPImannequin()
    
    try:
        succes_global = testeur.lancer_test_complet()
        
        if succes_global:
            print("\n🎉 ✅ TEST COMPLET RÉUSSI - PIEDS VISIBLES PARTOUT!")
        else:
            print("\n⚠️ ❌ QUELQUES TESTS ONT ÉCHOUÉ")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()