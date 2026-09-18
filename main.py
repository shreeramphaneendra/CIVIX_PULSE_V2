from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx
import math
from datetime import datetime, timezone, timedelta
import io
import json
import asyncio
from groq import AsyncGroq
import traceback
import base64
import re 
import tempfile
import shutil

# --- 🚀 OPEN SOURCE CLUSTERING LIBRARIES ---
from sentence_transformers import SentenceTransformer
import imagehash
from PIL import Image

import os
from dotenv import load_dotenv

# --- 🚀 NEW: DINOv2 GEOMETRIC ENGINE ---
from deduplication_engine import DeduplicationEngine

load_dotenv()

# ==========================================
# 🛑 CONFIGURATION (SECURED) 🛑
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
groq_client = AsyncGroq(api_key=GROQ_API_KEY, timeout=25.0)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

user_sessions = {}

print("Loading Local Embedding Model... (This takes a few seconds)")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded! System Ready.")

# --- 🚀 INITIALIZE ADVANCED DEDUPLICATION ENGINE ---
print("Loading DINOv2 & Vision Engine...")
dedup_engine = DeduplicationEngine(
    geo_radius_meters=75.0,
    inlier_threshold_merge=35,
    inlier_threshold_review=15
)
print("Vision Engine Loaded!")

# ==========================================
# 🛠️ TITANIUM EXTRACTOR (NUKES <THINK> TAGS)
# ==========================================
def extract_json_safely(raw_text):
    if not raw_text: return None
    
    # 1. Brutally rip out <think> blocks from reasoning models
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 2. Strip Markdown code blocks
    clean_text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    try: 
        return json.loads(clean_text)
    except Exception:
        # 3. Fallback: Find first { and last }
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return None

# 🔥 CUSTOM HACKATHON-TIER MODELS 🔥
TEXT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b", "groq/compound"]
VISION_MODELS = ["qwen/qwen3.6-27b"]

async def auto_route_groq_text(messages):
    for model in TEXT_MODELS:
        try:
            res = await asyncio.wait_for(groq_client.chat.completions.create(model=model, messages=messages), timeout=10.0)
            return res.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Auto-Router: Text Model '{model}' failed ({e}). Trying next...")
            continue
    print("❌ ALL TEXT MODELS FAILED.")
    return None

async def auto_route_groq_vision(prompt, image_url_or_base64):
    final_url = image_url_or_base64 if image_url_or_base64.startswith("http") else f"data:image/jpeg;base64,{image_url_or_base64}"
    for model in VISION_MODELS:
        try:
            res = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": final_url}}]}]
                ), timeout=15.0
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Auto-Router: Vision Model '{model}' failed ({e}). Trying next...")
            continue
    print("❌ ALL VISION MODELS FAILED.")
    return None

# ==========================================
# 📋 CUSTOM MENUS & CATEGORIES
# ==========================================
ISSUE_CATEGORIES = {
    "pothole":     {"label": "🕳️ Pothole",        "clean": "Pothole",         "category": "Roads"},
    "streetlight": {"label": "💡 Streetlight",     "clean": "Streetlight",     "category": "Electricity"},
    "water":       {"label": "🚰 Water Leak",      "clean": "Water Leak",      "category": "Water"},
    "manhole":     {"label": "⚠️ Open Manhole",    "clean": "Open Manhole",    "category": "Water"},
    "garbage":     {"label": "🗑️ Garbage Dump",    "clean": "Garbage Dump",    "category": "Sanitation"},
    "sewage":      {"label": "🌊 Sewage Overflow", "clean": "Sewage Overflow", "category": "Water"},
    "tree":        {"label": "🌳 Fallen Tree",     "clean": "Fallen Tree",     "category": "Public_Safety"},
    "footpath":    {"label": "🚧 Broken Footpath", "clean": "Broken Footpath", "category": "Roads"},
    "other":       {"label": "❓ Other Issue",     "clean": "Other Issue",     "category": "Other"},
}

def build_category_keyboard():
    labels = [info["label"] for info in ISSUE_CATEGORIES.values()]
    rows = [[{"text": label} for label in labels[i:i+2]] for i in range(0, len(labels), 2)]
    rows.append([{"text": "❌ Cancel"}])
    return {"keyboard": rows, "resize_keyboard": True}

def get_main_menu_keyboard():
    return {
        "keyboard": [
            [{"text": "🚨 Report a Grievance"}, {"text": "📍 View Nearby Issues"}],
            [{"text": "🤖 Chat with Civix AI"}, {"text": "❓ Help / Info"}]
        ],
        "resize_keyboard": True
    }

def get_cancel_keyboard():
    return {"keyboard": [[{"text": "❌ Cancel"}]], "resize_keyboard": True}

# ==========================================
# 🛡️ EXPLICIT NETWORK SENDER
# ==========================================
http_client = httpx.AsyncClient(timeout=20.0)

async def safe_request(method, url, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            if method == "GET": return await http_client.get(url, **kwargs)
            elif method == "POST": return await http_client.post(url, **kwargs)
            elif method == "PATCH": return await http_client.patch(url, **kwargs)
        except Exception as e:
            print(f"⚠️ [NETWORK EXCEPTION] Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1: return None 
            await asyncio.sleep(1.0) 

async def send_message(chat_id, text, reply_markup=None, use_markdown=False):
    if not chat_id: return
    if not text: text = "Processing..."
    
    print(f"✉️ [NETWORK] Sending to {chat_id}: '{text[:40]}...'")
    
    payload = {"chat_id": str(chat_id), "text": str(text)}
    if use_markdown: payload["parse_mode"] = "Markdown"
    if reply_markup: payload["reply_markup"] = reply_markup
        
    res = await safe_request("POST", f"{TELEGRAM_API_URL}/sendMessage", json=payload)
    
    if res is None:
        print(f"❌ [NETWORK CRASH] Could not reach Telegram API.")
        return None
        
    if res.status_code != 200:
        print(f"⚠️ [WARNING] Telegram rejected message: {res.text}. Retrying plain text...")
        payload.pop("parse_mode", None)
        payload.pop("reply_markup", None) 
        res = await safe_request("POST", f"{TELEGRAM_API_URL}/sendMessage", json=payload)
        if res and res.status_code == 200:
            print(f"✅ [SUCCESS] Delivered via Plain Text Fallback.")
        else:
            print(f"❌ [FATAL] Telegram completely rejected message: {res.text if res else 'None'}")
    else:
        print(f"✅ [SUCCESS] Delivered perfectly.")
        
    return res

async def get_telegram_image_bytes(file_id):
    res1 = await safe_request("GET", f"{TELEGRAM_API_URL}/getFile?file_id={file_id}")
    if not res1 or res1.status_code != 200: return None
    file_path = res1.json()['result']['file_path']
    res2 = await safe_request("GET", f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}")
    return res2.content if res2 and res2.status_code == 200 else None

# ==========================================
# 🧠 AI AGENTS
# ==========================================
async def agent_civix_chat(user_text):
    prompt = """You are Civix, a professional smart city AI. Answer civic/infrastructure questions factually. Keep it brief. Do not ask open ended questions."""
    raw_output = await auto_route_groq_text([{"role": "system", "content": prompt}, {"role": "user", "content": user_text}])
    if not raw_output: return "⚠️ The AI servers are currently down. Please try again later."
    return raw_output

async def agent_summarize_complaint(raw_text):
    prompt = "Read this civic complaint and output a strictly 1-line, professional 8-10 word summary. No quotes, no formatting."
    raw_output = await auto_route_groq_text([{"role": "system", "content": prompt}, {"role": "user", "content": raw_text}])
    if not raw_output: return (raw_text[:50] + '...') if len(raw_text) > 50 else raw_text
    return extract_json_safely(raw_output) or raw_output.strip().replace('"', '')

async def transcribe_voice(file_id):
    res1 = await safe_request("GET", f"{TELEGRAM_API_URL}/getFile?file_id={file_id}")
    if not res1 or res1.status_code != 200: return None
    res2 = await safe_request("GET", f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{res1.json()['result']['file_path']}")
    if not res2 or res2.status_code != 200: return None
    try:
        transcription = await groq_client.audio.transcriptions.create(file=("audio.ogg", res2.content), model="whisper-large-v3-turbo")
        return transcription.text
    except Exception:
        try:
            transcription = await groq_client.audio.transcriptions.create(file=("audio.ogg", res2.content), model="whisper-large-v3")
            return transcription.text
        except: return None

async def agent_filter(user_text):
    print("🧠 AGENT 1 (Filter): Analyzing text intent...")
    prompt = """
    You are a strict Civic Grievance intent analyzer. 
    If the user is greeting you (e.g., "hello", "hey") or chatting, set "is_complaint": false and write a polite "bot_reply" asking them to describe their infrastructure issue.
    If they are describing a physical problem (pothole, leak, outage, trash, broken street), set "is_complaint": true and leave "bot_reply" blank.
    
    Output ONLY a valid JSON object. No markdown. Example: {"is_complaint": true, "bot_reply": ""}
    """
    raw_output = await auto_route_groq_text([{"role": "system", "content": prompt}, {"role": "user", "content": user_text}])
    
    if not raw_output:
        return {"is_complaint": False, "bot_reply": "⚠️ The AI is currently offline. Cannot process text right now."}
        
    parsed = extract_json_safely(raw_output)
    return parsed if parsed else {"is_complaint": False, "bot_reply": "⚠️ AI format error. Please try describing the issue again."}

async def agent_visual_auditor(image_bytes, complaint_text):
    print("👁️ AGENT 4 (Vision): Auditing Image...")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB": img = img.convert("RGB")
        img.thumbnail((800, 800)) 
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        optimized_bytes = buf.getvalue()
    except Exception: optimized_bytes = image_bytes
    base64_image = base64.b64encode(optimized_bytes).decode('utf-8')

    prompt = f"""
    You are a strict municipal image verifier. Output ONLY a valid JSON object.
    1. Check if the physical issue in the image matches: "{complaint_text}".
    2. SPOOF DETECTION: Set "is_real": false if it is a cartoon, drawing, or manipulated.
    3. Extract any readable text (OCR).
    
    Format:
    {{
        "is_relevant": true,
        "is_real": true,
        "ocr_text": "text or 'None'",
        "reason": "Brief explanation"
    }}
    
    DO NOT use Markdown format. DO NOT use backticks. Just output the raw JSON object.
    """
    raw_output = await auto_route_groq_vision(prompt, base64_image)
    print(f"\n🤖 [VISION RAW OUTPUT]:\n{raw_output}\n")
    
    if not raw_output:
        return {"is_relevant": False, "is_real": False, "ocr_text": "None", "reason": "⚠️ Vision AI is offline. Cannot verify image."}
        
    parsed_data = extract_json_safely(raw_output)
    if parsed_data: return parsed_data
    return {"is_relevant": False, "is_real": False, "ocr_text": "None", "reason": "System error parsing the AI vision response."}

async def agent_triage(complaint_text, ocr_text="", user_category=""):
    print("🧠 AGENT 2 (Triage): Running Matrix...")
    prompt = f"""
    Score (0-100) based on: 1. Safety Risk 2. Infrastructure Damage 3. Community Impact.
    Issue: "{complaint_text}". Category: {user_category}.
    >=80: CRITICAL | 50-79: HIGH | <50: LOW
    Output ONLY JSON:
    {{"is_legit": true, "category": "string", "priority_score": 85, "priority_level": "CRITICAL", "cluster_tag": "word", "reasoning": "brief"}}
    """
    raw_output = await auto_route_groq_text([{"role": "user", "content": prompt}])
    if not raw_output: return {"is_legit": True, "category": user_category, "priority_level": "HIGH", "cluster_tag": "issue"}
    parsed = extract_json_safely(raw_output)
    return parsed if parsed else {"is_legit": True, "category": user_category, "priority_level": "HIGH", "cluster_tag": "issue"}

# ==========================================
# 🧠 DINOv2 CLUSTERING & UTILS (UPDATED)
# ==========================================
async def run_clustering_pipeline(lat, lng, complaint_text, file_id, db_headers):
    print("\n🔍 Running DINOv2 Advanced Clustering Pipeline...")
    query_embedding = []
    img_hash_str = "DINOv2_Vector_Processed"

    try:
        # 1. Get Text Embedding (Fallback included)
        try: query_embedding = embedding_model.encode(complaint_text).tolist()
        except: query_embedding = [0.0] * 384 

        # 2. Get incoming image from Telegram
        img_bytes = await get_telegram_image_bytes(file_id)
        if not img_bytes:
            return None, query_embedding, img_hash_str

        # 3. Create temp directory for Vision processing
        temp_dir = tempfile.mkdtemp()
        query_img_path = os.path.join(temp_dir, f"query_{file_id}.jpg")
        with open(query_img_path, "wb") as f:
            f.write(img_bytes)

        # 4. Fetch active tickets from Supabase for Geo-Fencing
        res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?status=in.(Open,Merged)&select=*", headers=db_headers)
        
        candidates = []
        if res and res.status_code == 200:
            raw_tickets = res.json()
            for t in raw_tickets:
                if not t.get('lat') or not t.get('lng'): continue
                
                # Check distance before downloading reference image (using existing haversine)
                dist = haversine(lat, lng, t['lat'], t['lng'])
                if dist <= 0.075: # 75 meters
                    ref_url = t.get('citizen_image_url')
                    if not ref_url: continue
                    
                    ref_img_path = os.path.join(temp_dir, f"ref_{t['id']}.jpg")
                    try:
                        # Safely download reference image using existing http_client logic
                        res_img = await safe_request("GET", ref_url)
                        if res_img and res_img.status_code == 200:
                            with open(ref_img_path, "wb") as f_ref:
                                f_ref.write(res_img.content)
                        else: continue
                    except Exception as e:
                        print(f"⚠️ Failed to cache ref image {t['id']}: {e}")
                        continue
                    
                    candidates.append({
                        "id": t["id"],
                        "lat": t["lat"],
                        "lng": t["lng"],
                        "local_image_path": ref_img_path,
                        "embedding": t.get("text_embedding"), 
                        "root_ticket_id": t.get("cluster_id")
                    })

        # 5. Run the new Vision Engine
        decision = dedup_engine.evaluate_incoming_report(
            new_image_path=query_img_path,
            new_lat=lat,
            new_lng=lng,
            existing_tickets=candidates
        )

        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)

        # 6. Interpret Decision for your existing State Machine
        if decision["action"] == "AUTO_MERGE" or decision["action"] == "FLAG_FOR_REVIEW":
            print(f"🚨 DUPLICATE CONFIRMED! Action: {decision['action']} (Inliers: {decision.get('inliers')})")
            
            matched_id = decision.get("matched_ticket_id")
            for t in raw_tickets:
                if t['id'] == matched_id:
                    return t, query_embedding, img_hash_str

        return None, query_embedding, img_hash_str

    except Exception as e:
        print(f"❌ Clustering Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        return None, query_embedding if query_embedding else [0.0] * 384, img_hash_str

def calculate_remaining_eta(created_at_iso, priority):
    created_time = datetime.fromisoformat(created_at_iso.replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    target_time = created_time + timedelta(hours=2 if priority == "CRITICAL" else 24 if priority == "HIGH" else 48)
    remaining = target_time - now
    if remaining.total_seconds() <= 0: return "Overdue"
    hours, remainder = divmod(remaining.seconds, 3600)
    return f"{remaining.days}d {hours}h" if remaining.days > 0 else f"{hours}h {remainder//60}m"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

async def get_nearest_zone(lat, lng, db_headers):
    res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/locations", headers=db_headers)
    if not res or res.status_code != 200: return None, "Unknown"
    closest_loc, min_dist = None, float('inf')
    for loc in res.json():
        dist = haversine(lat, lng, loc['center_lat'], loc['center_lng'])
        if dist < min_dist: min_dist, closest_loc = dist, loc
    return closest_loc['id'], closest_loc['name']

# ==========================================
# 👷 AGENT 5: WORKER RESOLUTION AUDITOR
# ==========================================
async def agent_worker_verifier(image_url, original_complaint):
    prompt = f"""
    You are auditing a civic worker's repair job. Output ONLY a valid JSON object.
    Original complaint: "{original_complaint}"
    Does it appear FIXED or RESOLVED?
    JSON Template: {{"is_resolved": true, "reason": "Brief explanation."}}
    """
    try:
        raw_output = await auto_route_groq_vision(prompt, image_url)
        if not raw_output: return {"is_resolved": True, "reason": "Bypassed due to AI failure."}
        parsed_data = extract_json_safely(raw_output)
        if parsed_data: return parsed_data
        raise Exception("Failed to parse")
    except Exception:
        return {"is_resolved": True, "reason": "Bypassed due to API limits."}

# ==========================================
# 🚀 MASS RESOLUTION & BROADCAST (WEBHOOK)
# ==========================================
class ResolutionData(BaseModel):
    task_id: str 
    resolution_image_url: str
    worker_id: str

@app.post("/agent-workflow/verify-resolution")
async def trigger_agent_3(data: ResolutionData):
    db_headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    t_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{data.task_id}", headers=db_headers)
    if not t_res or not t_res.json(): return {"status": "error"}
    target_ticket = t_res.json()[0]

    verification = await agent_worker_verifier(data.resolution_image_url, target_ticket['original_complaint'])
    if not verification.get("is_resolved", True):
        return {"status": "rejected", "reason": "AI Auditor rejected the repair photo. Please upload a clear photo of the resolved issue."}
    
    tickets_to_resolve = [target_ticket]
    cluster_id = target_ticket.get('cluster_id') or data.task_id
    c_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?cluster_id=eq.{cluster_id}&status=eq.Merged", headers=db_headers)
    if c_res and c_res.status_code == 200 and c_res.json():
        for child in c_res.json():
            if child['id'] != target_ticket['id']: tickets_to_resolve.append(child)

    for t in tickets_to_resolve:
        cit_id = t['citizen_chat_id']
        caption = f"✅ Civix-Pulse Update\nThe grievance in your area is officially RESOLVED!\nAudit Status: {verification.get('reason')}"
        photo_res = await safe_request("POST", f"{TELEGRAM_API_URL}/sendPhoto", json={"chat_id": str(cit_id), "photo": data.resolution_image_url, "caption": caption})
        if not photo_res or photo_res.status_code != 200:
            await send_message(cit_id, f"✅ Civix-Pulse Update\nYour reported issue is RESOLVED! \nView the repair photo here: {data.resolution_image_url}")
        await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{t['id']}", headers=db_headers, json={"status": "Resolved", "resolution_image_url": data.resolution_image_url})

    worker_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/workers?id=eq.{data.worker_id}&select=*", headers=db_headers)
    worker_data = worker_res.json()[0] if worker_res and worker_res.status_code == 200 and worker_res.json() else None

    if worker_data:
        cat, loc_id = worker_data['skill_category'], worker_data['location_id']
        queue_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?status=eq.Open&assigned_worker=is.null&category=eq.{cat}&location_id=eq.{loc_id}&order=created_at.asc&limit=1", headers=db_headers)
        pending_tasks = queue_res.json() if queue_res and queue_res.status_code == 200 else []

        if pending_tasks:
            next_task = pending_tasks[0]
            await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{next_task['id']}", headers=db_headers, json={"assigned_worker": data.worker_id})
            await send_message(next_task['citizen_chat_id'], f"👷 DISPATCH UPDATE\nA unit has finished their previous job and is now en route to resolve your ticket!")
            worker_telegram = worker_data.get('telegram_chat_id') or worker_data.get('chat_id')
            if worker_telegram:
                await send_message(worker_telegram, f"🚨 QUEUED DISPATCH 🚨\nPriority: {next_task['priority_level']}\nCheck your portal.")
        else:
            await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/workers?id=eq.{data.worker_id}", headers=db_headers, json={"status": "Available"})

    return {"status": "success"}

# ==========================================
# 🛡️ THE STRICT SEQUENTIAL STATE MACHINE
# ==========================================
async def reset_to_main_menu(chat_id, message="Menu reset."):
    user_sessions[chat_id] = {
        "step": "main_menu", "category": None, "complaint_text": "", 
        "photo_id": None, "ocr_text": "", "lat": None, "lng": None
    }
    await send_message(chat_id, message, reply_markup=get_main_menu_keyboard(), use_markdown=True)

async def process_telegram_update(message):
    try:
        chat_id = message.get("chat", {}).get("id")
        if not chat_id: return

        raw_text = message.get("text") or message.get("caption") or ""
        user_text = str(raw_text).lower().strip()

        print(f"\n🗣️ USER ({chat_id}) SAID: '{user_text}' | Data keys sent: {list(message.keys())}")

        GLOBAL_COMMANDS = ["/start", "cancel", "reset", "restart", "menu", "❌ cancel"]
        MAIN_MENU_TRIGGERS = ["report a grievance", "chat with civix ai", "view nearby issues", "help / info"]

        if user_text in GLOBAL_COMMANDS:
            await reset_to_main_menu(chat_id, "Welcome to *Civix-Pulse*! 🏙️\nSelect an option below to get started:")
            return
            
        if chat_id not in user_sessions: 
            await reset_to_main_menu(chat_id, "Session started. Please select an option:")
            return

        session = user_sessions[chat_id]

        if any(t in user_text for t in MAIN_MENU_TRIGGERS):
            session["step"] = "main_menu"

        # ----------------------------------------
        # MAIN MENU ROUTING
        # ----------------------------------------
        if session["step"] == "main_menu":
            if "report a grievance" in user_text:
                session["step"] = "category_selection"
                await send_message(chat_id, "What type of issue are you reporting?", reply_markup=build_category_keyboard())
            elif "chat with civix ai" in user_text:
                session["step"] = "chatting"
                await send_message(chat_id, "🤖 *Civix AI Online*\nAsk me about city operations or infrastructure.", reply_markup=get_cancel_keyboard(), use_markdown=True)
            elif "view nearby issues" in user_text:
                session["step"] = "waiting_for_radar_location"
                await send_message(chat_id, "📍 *Nearby Radar Active*\nPlease share your Live Location.", reply_markup=get_cancel_keyboard(), use_markdown=True)
            elif "help / info" in user_text:
                await send_message(chat_id, "🏛️ *Civix-Pulse AI*\nWe analyze, cluster, and route civic issues using Groq AI.", reply_markup=get_main_menu_keyboard(), use_markdown=True)
            else:
                # 🔥 AI CHAT AT MAIN MENU
                await send_message(chat_id, "🤔 Thinking...", use_markdown=False)
                ai_reply = await agent_civix_chat(raw_text)
                await send_message(chat_id, ai_reply, reply_markup=get_main_menu_keyboard())
            return

        # ----------------------------------------
        # EXTRA FEATURES
        # ----------------------------------------
        if session["step"] == "chatting":
            await send_message(chat_id, "🤔 Analyzing...", use_markdown=True)
            ai_reply = await agent_civix_chat(raw_text) 
            await send_message(chat_id, ai_reply, reply_markup=get_cancel_keyboard())
            return

        if session["step"] == "waiting_for_radar_location":
            if "location" in message:
                lat, lng = message["location"]["latitude"], message["location"]["longitude"]
                await send_message(chat_id, "📡 Scanning area...", use_markdown=True)
                db_headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
                t_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?status=in.(Open,Merged)&select=category,priority_level,extracted_location,original_complaint,lat,lng", headers=db_headers)
                if t_res and t_res.status_code == 200:
                    tickets = t_res.json()
                    nearby = [{"dist": haversine(lat, lng, t['lat'], t['lng']), "cat": t['category'], "pri": t['priority_level'], "loc": t.get('extracted_location',''), "desc": t.get('original_complaint', '')} for t in tickets if t.get('lat')]
                    nearby = [n for n in nearby if n['dist'] < 5.0]
                    if nearby:
                        nearby.sort(key=lambda x: x['dist'])
                        summaries = await asyncio.gather(*[agent_summarize_complaint(issue['desc']) for issue in nearby[:3]])
                        msg = "📍 *Nearby Issues:*\n\n"
                        for i, (issue, summary) in enumerate(zip(nearby[:3], summaries)):
                            msg += f"{i+1}. {issue['cat']} ({issue['dist']:.2f}km away)\n   📝 {summary}\n\n"
                        await reset_to_main_menu(chat_id, msg)
                    else:
                        await reset_to_main_menu(chat_id, "✅ No active issues within 5km.")
                else: await reset_to_main_menu(chat_id, "⚠️ Database error.")
            else: await send_message(chat_id, "Please share a Location Pin or tap Cancel.")
            return

        # ----------------------------------------
        # STRICT STEP 1: CATEGORY
        # ----------------------------------------
        if session["step"] == "category_selection":
            selected_category = next((info["category"] for info in ISSUE_CATEGORIES.values() if info["clean"].lower() in user_text or info["label"].lower() in user_text), None)
            if selected_category:
                session["category"] = selected_category
                session["step"] = "waiting_for_text"
                await send_message(chat_id, f"✅ Category set to {selected_category}.\n\n👉 *Step 1: Please type or speak a brief description of the issue.*", reply_markup=get_cancel_keyboard(), use_markdown=True)
            else:
                await send_message(chat_id, "Please select a category from the buttons.")
            return

        # ----------------------------------------
        # STRICT STEP 2: TEXT/VOICE (WITH HONEST FILTER)
        # ----------------------------------------
        if session["step"] == "waiting_for_text":
            if "photo" in message or "location" in message:
                await send_message(chat_id, "Hold on! I need the text/voice description first. Please describe the issue.", reply_markup=get_cancel_keyboard())
                return

            if "voice" in message:
                await send_message(chat_id, "🎙️ Transcribing...")
                text = await transcribe_voice(message["voice"]["file_id"])
                if text:
                    await send_message(chat_id, f"📝 I heard: '{text}'")
                    raw_text = text
                else:
                    await send_message(chat_id, "Could not transcribe. Please type it.")
                    return

            if raw_text:
                analysis = await agent_filter(raw_text)
                if analysis.get("is_complaint"):
                    session["complaint_text"] = raw_text
                    session["step"] = "waiting_for_photo"
                    await send_message(chat_id, "✅ Description saved.\n\n👉 *Step 2: Please upload a clear photo of the issue.*", reply_markup=get_cancel_keyboard(), use_markdown=True)
                else:
                    bot_reply = analysis.get("bot_reply", "Please describe a valid infrastructure issue.")
                    await send_message(chat_id, bot_reply, reply_markup=get_cancel_keyboard())
            return

        # ----------------------------------------
        # STRICT STEP 3: PHOTO (WITH HONEST VISION AI)
        # ----------------------------------------
        if session["step"] == "waiting_for_photo":
            if "location" in message or raw_text:
                await send_message(chat_id, "I am waiting for a Photo! Please upload an image using the attachment icon.", reply_markup=get_cancel_keyboard())
                return

            photo_id = None
            if "photo" in message and len(message["photo"]) > 0:
                photo_id = message["photo"][-1].get("file_id")
            elif "document" in message:
                photo_id = message["document"].get("file_id")

            if photo_id:
                await send_message(chat_id, "👁️ Analyzing your photo...", reply_markup=get_cancel_keyboard())
                img_bytes = await get_telegram_image_bytes(photo_id)
                if img_bytes:
                    vision_analysis = await agent_visual_auditor(img_bytes, session['complaint_text'])
                    if not vision_analysis.get("is_relevant") or not vision_analysis.get("is_real"):
                        await send_message(chat_id, f"❌ Photo Rejected.\nReason: {vision_analysis.get('reason')}\n\nPlease upload a valid photo.", reply_markup=get_cancel_keyboard())
                    else:
                        session["photo_id"] = photo_id
                        session["ocr_text"] = vision_analysis.get("ocr_text", "")
                        session["step"] = "waiting_for_location"
                        await send_message(chat_id, "✅ Photo verified!\n\n👉 *Final Step: Please share your Live Location Pin.*", reply_markup=get_cancel_keyboard(), use_markdown=True)
                else:
                    await send_message(chat_id, "❌ Error downloading image.")
            return

        # ----------------------------------------
        # STRICT STEP 4: LOCATION & DISPATCH
        # ----------------------------------------
        if session["step"] == "waiting_for_location":
            if "location" not in message:
                await send_message(chat_id, "Almost done! Please tap the attachment icon and share your Location.", reply_markup=get_cancel_keyboard())
                return
                
            session["lat"] = message["location"]["latitude"]
            session["lng"] = message["location"]["longitude"]
            await send_message(chat_id, "✅ Location saved! Initiating AI Analysis & Clustering... ⏳")
            
            db_headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
            loc_id, loc_name = await get_nearest_zone(session["lat"], session["lng"], db_headers)
            
            duplicate_ticket, vec_emb, img_hash = await run_clustering_pipeline(session["lat"], session["lng"], session['complaint_text'], session['photo_id'], db_headers)
            
            if duplicate_ticket:
                parent_id = duplicate_ticket['id']
                p_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{parent_id}", headers=db_headers)
                parent_data = p_res.json()[0] if p_res and p_res.json() else None
                
                if parent_data:
                    ticket_data = {
                        "citizen_chat_id": str(chat_id), "original_complaint": session['complaint_text'],
                        "category": parent_data['category'], "priority_level": parent_data['priority_level'], 
                        "cluster_id": parent_data.get('cluster_id') or parent_id, "extracted_location": loc_name, "location_id": loc_id, 
                        "status": "Merged", "lat": session["lat"], "lng": session["lng"], "citizen_image_url": session['photo_id'], 
                        "assigned_worker": parent_data['assigned_worker'], "text_embedding": vec_emb, "image_hash": img_hash
                    }
                    await safe_request("POST", f"{SUPABASE_URL}/rest/v1/grievances", headers=db_headers, json=ticket_data)

                worker_name = duplicate_ticket.get('workers', {}).get('name', 'an active unit')
                eta_remaining = calculate_remaining_eta(parent_data['created_at'], parent_data['priority_level']) if parent_data else "24 Hours"
                msg = f"🚨 *Existing Issue Detected!*\n\nOur mapping confirms this issue is already grouped with a verified cluster.\n👷 Unit: {worker_name}\n⏱️ ETA: {eta_remaining}"
                await reset_to_main_menu(chat_id, msg)
            else: 
                ai_result = await agent_triage(session['complaint_text'], session['ocr_text'], session['category'])
                if ai_result and ai_result.get("is_legit"):
                    category, priority = ai_result.get('category', 'Other'), ai_result.get('priority_level', 'HIGH')
                    eta = "48 Hours" if priority == "LOW" else "24 Hours" if priority == "HIGH" else "2 Hours"
                    
                    assigned_worker_id, worker_name = None, "Pending Assignment"
                    w_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/workers?skill_category=eq.{category}&status=eq.Available&location_id=eq.{loc_id}&limit=1", headers=db_headers)
                    if w_res and w_res.status_code == 200 and len(w_res.json()) > 0:
                        worker = w_res.json()[0]
                        assigned_worker_id = worker.get('id')
                        worker_name = worker.get('name', 'Unit Dispatched')
                        
                        await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/workers?id=eq.{assigned_worker_id}", headers=db_headers, json={"status": "Dispatched"})
                        if worker.get('telegram_chat_id'):
                            await send_message(worker.get('telegram_chat_id'), f"🚨 NEW DISPATCH 🚨\nZone: {loc_name}\nCategory: {category}\nPriority: {priority}\nETA: {eta}")

                    ticket_data = {
                        "citizen_chat_id": str(chat_id), "original_complaint": session['complaint_text'],
                        "category": category, "priority_level": priority, "cluster_id": ai_result.get('cluster_tag'),
                        "extracted_location": loc_name, "location_id": loc_id, "status": "Open",
                        "lat": session["lat"], "lng": session["lng"], "citizen_image_url": session['photo_id'], "assigned_worker": assigned_worker_id,
                        "text_embedding": vec_emb, "image_hash": img_hash
                    }
                    
                    db_res = await safe_request("POST", f"{SUPABASE_URL}/rest/v1/grievances", headers=db_headers, json=ticket_data)
                    if db_res and db_res.status_code in [200, 201]:
                        msg = f"✅ *Official Ticket Created!*\n📍 Zone: {loc_name}\n📋 Category: {category}\n🚨 Matrix Priority: {priority}\n👷 Unit: {worker_name}\n⏱️ Target ETA: {eta}"
                        await reset_to_main_menu(chat_id, msg)
                    else:
                        await reset_to_main_menu(chat_id, "⚠️ AI processed the issue, but the database rejected the ticket creation.")
                else: 
                    await reset_to_main_menu(chat_id, "❌ AI determined this is not a valid civic complaint.")

    except Exception as e:
        print(f"❌ [FATAL ERROR]: {e}")
        traceback.print_exc()
        try:
            chat_id = message.get("chat", {}).get("id")
            if chat_id: await reset_to_main_menu(chat_id, "⚠️ A glitch occurred. Session reset.")
        except: pass

@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    if "message" in data: 
        background_tasks.add_task(process_telegram_update, data["message"])
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)