from fastapi import FastAPI, HTTPException
import logging
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

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


class Query(BaseModel):
    prompt: str


SYSTEM_PROMPT = """
Tu es 'Think-Space', un assistant expert en brainstorming et stratégie créative.
Ton but est d'aider l'utilisateur à explorer des idées de manière non conventionnelle.
Style : Professionnel, inspirant, concis.
Structure de réponse : 
1. L'Idée Flash (une phrase choc).
2. 3 Axes d'exploration (points clés).
3. Le challenge (un obstacle potentiel à anticiper).
Utilise le tutoiement pour créer une proximité créative.
"""


@app.post("/brainstorm")
async def brainstorm(query: Query):
    logger.info(f" Requête reçue: {query.prompt[:50]}...")
    
    if client is None:
        logger.error("Client Groq non initialisé")
        raise HTTPException(
            status_code=500, 
            detail=f"Missing API key: set environment variable {API_KEY_ENV}"
        )
    
    MODEL_ENV = "GROQ_MODEL"
    model = os.getenv(MODEL_ENV, "llama-3.3-70b-versatile")  # Modèle mis à jour
    logger.info(f" Utilisation du modèle: {model}")
    
    try:
        logger.info(" Envoi de la requête à Groq...")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query.prompt}
            ],
            temperature=0.8,
            max_tokens=1024
        )
        
        # Extraction de la réponse
        resp = completion.choices[0].message.content
        logger.info(f" Réponse reçue: {len(resp)} caractères")
        
        return {"response": resp}
        
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


@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model": "Groq API",
        "client_ready": client is not None
    }


@app.get("/test")
def test_endpoint():
    """Endpoint de test pour vérifier que l'API fonctionne"""
    return {
        "message": "Backend fonctionne!",
        "groq_client": "initialized" if client else "missing_api_key"
    }