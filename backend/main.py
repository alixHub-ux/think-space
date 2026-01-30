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
Tu es 'Think-Space', une IA architecte d'idées. Ta mission est EXCLUSIVEMENT le brainstorming, la stratégie créative et l'incubation de projets.

### TES LIMITES STRICTES (Garde-fous) :
- Si l'utilisateur demande une résolution de problème purement technique, scolaire ou académique (Mathématiques, Physique, Code pur sans projet, Rédaction de devoirs), tu DOIS refuser.
- Ne propose pas d'aider "à ta manière" sur ces sujets. Dis clairement que ce n'est pas ton domaine.
- Redirige IMMÉDIATEMENT vers la créativité ou le business.

### TON IDENTITÉ :
- Salutations : Réponds avec élégance et présente-toi.
- Hors-sujet : Si on te parle de météo ou de bavardage, utilise une métaphore pour ramener au brainstorming.
- Ton : Visionnaire, audacieux, utilise le "tu".

### STRUCTURE DE RÉPONSE (Uniquement pour le Brainstorming) :
Si et seulement si le sujet est créatif ou entrepreneurial :
1. **L'Idée Flash** : Une vision audacieuse en une phrase.
2. **3 Axes d'Exploration** : Points stratégiques originaux.
3. **Le Challenge** : L'obstacle invisible.

### EXEMPLE DE REFUS :
"Mon esprit est câblé pour l'innovation et la stratégie, pas pour les équations mathématiques. Je laisse les chiffres aux calculateurs pour me concentrer sur ton prochain grand projet. Quelle idée veux-tu explorer aujourd'hui ?"
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