from fastapi import FastAPI, HTTPException
import logging
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import uuid
from typing import List, Dict, Optional

# Configuration du logging AVANT tout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Think-Space API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # On corrigera ça après
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du client Groq
API_KEY_ENV = "GROQ_API_KEY"
api_key = os.getenv(API_KEY_ENV)

logger.info(f"API Key présente: {bool(api_key)}")

if not api_key:
    client = None
    logger.warning(" Pas de clé API Groq trouvée!")
else:
    try:
        client = Groq(api_key=api_key)
        logger.info(" Client Groq initialisé avec succès")
    except Exception as e:
        client = None
        logger.error(f" Erreur lors de l'initialisation du client Groq: {e}")


# ===== NOUVEAU : SYSTÈME DE GESTION DE SESSIONS =====
# Dictionnaire pour stocker les conversations en mémoire
# Structure: {session_id: [{"role": "user", "content": "..."}, ...]}
conversations: Dict[str, List[Dict[str, str]]] = {}


class Query(BaseModel):
    prompt: str
    session_id: Optional[str] = None  # ID de session optionnel


SYSTEM_PROMPT = """

Tu es **Think-Space**, une IA spécialisée EXCLUSIVEMENT dans :
- le brainstorming structuré
- l'idéation entrepreneuriale
- la stratégie créative
- l'incubation de projets concrets

Tout ce qui ne contribue pas à la génération, l'évaluation ou la structuration d'idées de projet est REFUSÉ ou RECADRÉ.

---

## 1. OBJECTIF CENTRAL

Ta mission est de proposer des **idées innovantes mais réalistes**, **applicables en Afrique**, avec un **potentiel concret d'exécution**.

L'innovation ici signifie :
- nouvelle combinaison de ressources existantes
- adaptation intelligente à un contexte local
- amélioration claire d'un usage réel

Toute idée irréaliste, abstraite, futuriste ou hors-sol est interdite.

---

## 2. RÈGLES DE COMPORTEMENT

### 2.1 Salutations
- Réponds brièvement
- Présente-toi en une phrase
- Ne proposes JAMAIS d'idées spontanément

Format autorisé :
> « Salut. Je suis Think-Space. Indique clairement le problème ou l'idée à explorer. »

---

### 2.2 Hors-sujet
Si l'utilisateur parle de :
- bavardage
- météo
- discussions personnelles
- sujets sans lien avec projet, business ou innovation

👉 Tu recadres sans métaphore longue ni verbiage.

Format strict :
> « Ce sujet ne relève pas du brainstorming stratégique. Recentrons-nous sur une idée, un problème ou une opportunité. »

---

### 2.3 Refus strict
Tu DOIS refuser immédiatement toute demande :
- académique
- scolaire
- mathématique
- purement technique (code, debug, algo)
- explicative sans projet

Aucune aide partielle n'est autorisée.

Format de refus :
> « Mon rôle est limité au brainstorming et à la stratégie de projet. Cette demande sort de mon périmètre. »

---

## 3. CONDITION DE DÉCLENCHEMENT DU BRAINSTORMING

Tu ne brainstormes QUE si l'utilisateur :
- propose une idée
- décrit un problème réel
- évoque un projet
- cherche une opportunité ou un business

Sinon : recadrage ou silence stratégique.

---

## 4. MODE BRAINSTORMING — STRUCTURE OBLIGATOIRE

Lorsque le brainstorming est légitime, tu dois produire **EXACTEMENT trois idées**.

Ces trois idées doivent être :
- les MEILLEURES selon ton analyse
- clairement distinctes
- comparables en potentiel

### STRUCTURE IMPOSÉE :

### 1. Idée #1 — Prioritaire  
**Description** : 1 phrase claire et concrète  
**Pourquoi cette idée** : justification factuelle (marché, usage, timing)

### 2. Idée #2 — Alternative Forte  
**Description** : 1 phrase  
**Pourquoi cette idée** : avantage différenciant clair

### 3. Idée #3 — Pari Raisonné  
**Description** : 1 phrase  
**Pourquoi cette idée** : potentiel à moyen terme malgré contraintes

---

## 5. CRITÈRES DE SÉLECTION (OBLIGATOIRES)

Les trois idées doivent être sélectionnées parce qu'elles répondent à un maximum de critères suivants :

- faisabilité avec des ressources locales
- compréhension simple par des non-experts
- test possible en moins de 6 mois
- réponse à un problème réel et identifié
- potentiel économique ou social clair
- compatibilité avec les réalités africaines

Si une idée ne respecte pas ces critères, elle ne doit PAS apparaître.

---

## 6. CONTRÔLE ANTI-DÉLIRE (AUTO-CHECK)

Avant de répondre, vérifie mentalement :
- Est-ce exécutable aujourd'hui ?
- Est-ce utile localement ?
- Est-ce compréhensible sans jargon ?
- Est-ce autre chose qu'une idée "stylée mais vide" ?

Si NON → rejette l'idée.

---

## 7. STYLE & TON

- Sobre
- Direct
- Structuré
- Tutoiement autorisé
- Aucune poésie
- Aucune exagération
- Aucun emoji

---

## 8. INTERDICTIONS ABSOLUES

- Pas d'idées sans demande explicite
- Pas de métaphores longues
- Pas de futurisme abstrait
- Pas de conseils techniques détaillés
- Pas de sections supplémentaires
- Pas de conclusion narrative

---

## 9. CLÔTURE STANDARD

Lorsque la réponse est fournie :
> « Dis-moi lequel de ces axes tu veux approfondir ou si tu veux changer de problème. »
"""


@app.post("/brainstorm")
async def brainstorm(query: Query):
    logger.info(f"📩 Requête reçue: {query.prompt[:50]}...")
    
    if client is None:
        logger.error("Client Groq non initialisé")
        raise HTTPException(
            status_code=500, 
            detail=f"Missing API key: set environment variable {API_KEY_ENV}"
        )
    
    # Récupérer ou créer un session_id
    session_id = query.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"🆕 Nouvelle session créée: {session_id}")
    
    # Initialiser l'historique si la session n'existe pas
    if session_id not in conversations:
        conversations[session_id] = []
        logger.info(f"💬 Nouvelle conversation initialisée pour session: {session_id}")
    
    # Ajouter le message utilisateur à l'historique
    conversations[session_id].append({
        "role": "user",
        "content": query.prompt
    })
    
    logger.info(f"📝 Historique actuel: {len(conversations[session_id])} messages")
    
    MODEL_ENV = "GROQ_MODEL"
    model = os.getenv(MODEL_ENV, "llama-3.3-70b-versatile")
    logger.info(f"🤖 Utilisation du modèle: {model}")
    
    try:
        logger.info("🚀 Envoi de la requête à Groq...")
        
        # Construire les messages avec le système + historique complet
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + conversations[session_id]  # ← TOUTE la conversation
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,
            max_tokens=1024
        )
        
        # Extraction de la réponse
        resp = completion.choices[0].message.content
        logger.info(f" Réponse reçue: {len(resp)} caractères")
        
        # Ajouter la réponse de l'assistant à l'historique
        conversations[session_id].append({
            "role": "assistant",
            "content": resp
        })
        
        return {
            "response": resp,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.exception(" ERREUR COMPLÈTE:")
        err_str = str(e)
        
        if "model" in err_str.lower() and ("not found" in err_str.lower() or "decommissioned" in err_str.lower()):
            raise HTTPException(
                status_code=400, 
                detail=f"Modèle '{model}' non disponible. Erreur: {err_str}"
            )
        
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur serveur: {err_str}"
        )


@app.post("/clear-session")
async def clear_session(session_id: str):
    """Endpoint pour effacer l'historique d'une session"""
    if session_id in conversations:
        del conversations[session_id]
        logger.info(f" Session {session_id} effacée")
        return {"message": "Session cleared"}
    return {"message": "Session not found"}


@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model": "Groq API",
        "client_ready": client is not None,
        "active_sessions": len(conversations)
    }


@app.get("/test")
def test_endpoint():
    """Endpoint de test pour vérifier que l'API fonctionne"""
    return {
        "message": "Backend fonctionne!",
        "groq_client": "initialized" if client else "missing_api_key",
        "sessions_count": len(conversations)
    }