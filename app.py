"""
ShopBot Backend v2
==================
• Loads/saves FAISS embeddings (no recompute on restart)
• Flask REST API serving the visual UI
• Real Amazon images + buy links in every response
• Integrates semantic search + NLP intent + store location
"""

import os, json, re, sqlite3, random, time, uuid, pickle
import numpy as np
import pandas as pd
#import torch
import faiss
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import warnings; warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)

# ── Paths ─────────────────────────────────────────────────
DATA_DIR    = "."
PRODUCTS_CSV  = os.path.join(DATA_DIR, "amazon_products.csv")
CATEGORIES_CSV= os.path.join(DATA_DIR, "amazon_categories.csv")
EMB_PATH      = "embeddings.npy"
IDX_PATH      = "faiss.index"
DF_PATH       = "products.parquet"
DB_PATH       = "shopbot.db"
MODEL_NAME    = "all-MiniLM-L6-v2"
SAMPLE_SIZE   = 50000

# ── Globals ───────────────────────────────────────────────
device     = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
model      = None
gpu_index  = None
df         = None
sessions   = {}          # in-memory session store

print(f"🖥️  Device: {device}")

# ══════════════════════════════════════════════════════════
#  STARTUP — load data, model, index
# ══════════════════════════════════════════════════════════

def load_system():
    global model, gpu_index, df

    # 1. Load / build DataFrame
    if os.path.exists(DF_PATH):
        print("📦 Loading cached DataFrame …")
        df = pd.read_parquet(DF_PATH)
    else:
        print("📦 Building DataFrame from CSVs …")
        prods = pd.read_csv(PRODUCTS_CSV)
        cats  = pd.read_csv(CATEGORIES_CSV)
        merged = prods.merge(cats, left_on="category_id", right_on="id", how="left")
        merged = merged.dropna(subset=["title"])
        merged = merged[merged["price"] > 0]
        merged = merged.sample(min(SAMPLE_SIZE, len(merged)), random_state=42).reset_index(drop=True)
        merged["listPrice"]        = merged["listPrice"].fillna(0)
        merged["stars"]            = merged["stars"].fillna(0)
        merged["reviews"]          = merged["reviews"].fillna(0).astype(int)
        merged["boughtInLastMonth"]= merged["boughtInLastMonth"].fillna(0).astype(int)
        merged["isBestSeller"]     = merged["isBestSeller"].fillna(False)
        merged["imgUrl"]           = merged["imgUrl"].fillna("")
        merged["productURL"]       = merged["productURL"].fillna("")
        merged["category_name"]    = merged["category_name"].fillna("General")
        df = merged
        df.to_parquet(DF_PATH)
        print(f"✅ DataFrame saved ({len(df):,} products)")

    # 2. Load SentenceTransformer
    print(f"🧠 Loading SentenceTransformer ({MODEL_NAME}) on {device} …")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 3. Load / build FAISS index
    if os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
        print("⚡ Loading cached embeddings + FAISS index …")
        embeddings = np.load(EMB_PATH)
        cpu_index  = faiss.read_index(IDX_PATH)
    else:
        print(f"🔢 Encoding {len(df):,} titles on {device} …")
        embeddings = model.encode(
            df["title"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cpu_index = faiss.IndexFlatIP(embeddings.shape[1])
        cpu_index.add(embeddings)
        np.save(EMB_PATH, embeddings)
        faiss.write_index(cpu_index, IDX_PATH)
        print("💾 Embeddings + index saved")

    # Push FAISS to GPU if available
    if device == "cuda":
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("🚀 FAISS index on GPU")
    else:
        gpu_index = cpu_index
        print("💻 FAISS index on CPU")

    # 4. Ensure SQLite sessions DB
    _init_db()
    print("✅ System ready!\n")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, user_msg TEXT, bot_msg TEXT,
            intent TEXT, sentiment TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS store_locations (
            category TEXT PRIMARY KEY,
            aisle TEXT, section TEXT, shelf TEXT, floor TEXT,
            map_x INTEGER, map_y INTEGER
        );
    """)
    # Seed store locations for all categories
    c.execute("SELECT COUNT(*) FROM store_locations")
    if c.fetchone()[0] == 0 and df is not None:
        cats = df["category_name"].unique()
        aisles   = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3","E1","E2"]
        sections = ["North Wing","South Wing","East Hall","West Hall","Central Area"]
        shelves  = ["Shelf 1","Shelf 2","Shelf 3","End Cap","Display Stand","Wall Mount"]
        floors   = ["Ground Floor","First Floor","Second Floor"]
        random.seed(42)
        for cat in cats:
            c.execute("""INSERT OR IGNORE INTO store_locations VALUES (?,?,?,?,?,?,?)""", (
                str(cat),
                random.choice(aisles), random.choice(sections),
                random.choice(shelves), random.choice(floors),
                random.randint(5,95), random.randint(5,95)
            ))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════
#  NLP HELPERS
# ══════════════════════════════════════════════════════════

INTENT_RULES = [
    ({"where","aisle"},             "find_location"),
    ({"where","find"},              "find_location"),
    ({"which","aisle"},             "find_location"),
    ({"where","section"},           "find_location"),
    ({"navigate"},                  "find_location"),
    ({"located"},                   "find_location"),
    ({"map"},                       "find_location"),
    ({"compare"},                   "compare"),
    ({"vs"},                        "compare"),
    ({"versus"},                    "compare"),
    ({"difference","between"},      "compare"),
    ({"price"},                     "check_price"),
    ({"cost"},                      "check_price"),
    ({"how","much"},                "check_price"),
    ({"stock"},                     "check_stock"),
    ({"available"},                 "check_stock"),
    ({"in","stock"},                "check_stock"),
    ({"buy","now"},                 "add_to_cart"),
    ({"add","cart"},                "add_to_cart"),
    ({"purchase"},                  "add_to_cart"),
    ({"review"},                    "check_reviews"),
    ({"rating"},                    "check_reviews"),
    ({"recommend"},                 "recommend"),
    ({"suggest"},                   "recommend"),
    ({"best"},                      "recommend"),
    ({"top"},                       "recommend"),
    ({"discount"},                  "discount"),
    ({"sale"},                      "discount"),
    ({"offer"},                     "discount"),
    ({"deal"},                      "discount"),
    ({"return"},                    "return_policy"),
    ({"refund"},                    "return_policy"),
    ({"exchange"},                  "return_policy"),
    ({"track"},                     "track_order"),
    ({"order","status"},            "track_order"),
    ({"hello"},                     "greet"),
    ({"hi"},                        "greet"),
    ({"hey"},                       "greet"),
    ({"bye"},                       "farewell"),
    ({"goodbye"},                   "farewell"),
    ({"help"},                      "help"),
]

PRICE_RE = [
    (r'under\s+\$?(\d+)',        lambda m: {"max": float(m.group(1))}),
    (r'below\s+\$?(\d+)',        lambda m: {"max": float(m.group(1))}),
    (r'less\s+than\s+\$?(\d+)', lambda m: {"max": float(m.group(1))}),
    (r'up\s+to\s+\$?(\d+)',     lambda m: {"max": float(m.group(1))}),
    (r'max\s+\$?(\d+)',          lambda m: {"max": float(m.group(1))}),
    (r'over\s+\$?(\d+)',         lambda m: {"min": float(m.group(1))}),
    (r'above\s+\$?(\d+)',        lambda m: {"min": float(m.group(1))}),
    (r'\$?(\d+)\s*[-to]+\s*\$?(\d+)', lambda m: {"min": float(m.group(1)), "max": float(m.group(2))}),
    (r'around\s+\$?(\d+)',       lambda m: {"min": float(m.group(1))*0.8, "max": float(m.group(1))*1.2}),
    (r'budget\s+(?:of\s+)?\$?(\d+)', lambda m: {"max": float(m.group(1))}),
]

STOPS = {"i","me","my","show","find","get","want","need","looking","for","the","a",
         "an","is","are","do","does","this","that","it","them","these","can","you",
         "tell","about","more","please","ok","have","will","would","could","should",
         "any","some","all","to","in","and","or","of","on","at","by","with","from",
         "just","really","very","quite","much","many","give","let","see","put",
         "take","make","like","good","great","best","nice","cool","where","which",
         "how","what","when","who","suggest","recommend","buy","purchase","search",
         "help","use"}

def classify_intent(text: str) -> str:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    for rule_words, intent in INTENT_RULES:
        if rule_words.issubset(words):
            return intent
    return "search"

def extract_price(text: str) -> dict | None:
    tl = text.lower()
    for pattern, extractor in PRICE_RE:
        m = re.search(pattern, tl)
        if m:
            return extractor(m)
    return None

def extract_keywords(text: str) -> list:
    return [w for w in re.findall(r'\b[a-z][a-z0-9]+\b', text.lower())
            if w not in STOPS and len(w) > 2][:8]

def sentiment(text: str) -> dict:
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol > 0.05:   return {"label":"positive","emoji":"😊","polarity":round(pol,2)}
    elif pol < -0.05: return {"label":"negative","emoji":"😞","polarity":round(pol,2)}
    return {"label":"neutral","emoji":"😐","polarity":round(pol,2)}


# ══════════════════════════════════════════════════════════
#  SEMANTIC SEARCH (FAISS)
# ══════════════════════════════════════════════════════════

def semantic_search(query: str, top_k: int = 12, price_range: dict = None,
                    category_hint: str = None) -> list:
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = gpu_index.search(q_emb, 100)
    results = df.iloc[indices[0]].copy()
    results["_score"] = distances[0]

    # Price filter
    if price_range:
        if "max" in price_range:
            results = results[results["price"] <= price_range["max"]]
        if "min" in price_range:
            results = results[results["price"] >= price_range["min"]]

    # Category hint filter (soft — only apply if still has results)
    if category_hint and len(results) > top_k:
        cat_filtered = results[results["category_name"].str.contains(category_hint, case=False, na=False)]
        if len(cat_filtered) >= 3:
            results = cat_filtered

    # Rerank: semantic score + stars + popularity
    results = results[results["price"] > 0].copy()
    max_reviews = results["reviews"].max() or 1
    max_bought  = results["boughtInLastMonth"].max() or 1
    results["rank"] = (
        results["_score"]              * 0.45 +
        results["stars"] / 5           * 0.30 +
        results["reviews"] / max_reviews * 0.15 +
        results["boughtInLastMonth"] / max_bought * 0.10
    )
    results = results.sort_values("rank", ascending=False)
    return _to_records(results.head(top_k))


def _to_records(frame) -> list:
    """Convert DataFrame rows to clean dicts with image + buy link."""
    out = []
    for _, r in frame.iterrows():
        discount = 0
        if r.get("listPrice", 0) > r["price"] > 0:
            discount = round((1 - r["price"] / r["listPrice"]) * 100)
        out.append({
            "asin":        str(r.get("asin","")),
            "title":       str(r["title"]),
            "price":       round(float(r["price"]), 2),
            "list_price":  round(float(r.get("listPrice", 0) or 0), 2),
            "discount":    discount,
            "stars":       float(r.get("stars", 0) or 0),
            "reviews":     int(r.get("reviews", 0) or 0),
            "bought":      int(r.get("boughtInLastMonth", 0) or 0),
            "is_best_seller": bool(r.get("isBestSeller", False)),
            "category":    str(r.get("category_name", "")),
            "img_url":     str(r.get("imgUrl", "")),
            "buy_url":     str(r.get("productURL", "")),
        })
    return out


# ══════════════════════════════════════════════════════════
#  STORE LOCATION
# ══════════════════════════════════════════════════════════

def get_location(category: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM store_locations WHERE category LIKE ? LIMIT 1", (f"%{category}%",))
    row = c.fetchone(); conn.close()
    if not row: return None
    return {"category": row[0], "aisle": row[1], "section": row[2],
            "shelf": row[3], "floor": row[4], "map_x": row[5], "map_y": row[6]}


# ══════════════════════════════════════════════════════════
#  RESPONSE BUILDER
# ══════════════════════════════════════════════════════════

GREETS = [
    "Hey there! 👋 Welcome to ShopBot! I can find products with images, show prices, compare items, locate them in the store, and more. What are you looking for?",
    "Hello! 😊 I'm your AI shopping guide. I search 50,000+ real products with live images and Amazon buy links. What can I help you find today?",
    "Hi! 🛍️ Ask me anything — product search, price checks, store location, comparisons — I've got it all. What do you need?",
]

def build_response(intent: str, query: str, session: dict) -> dict:
    price_range   = extract_price(query)
    kws           = extract_keywords(query)
    products      = []
    location      = None
    escalate      = False
    response_type = intent

    # ── GREET ─────────────────────────────────────────────
    if intent == "greet":
        text = random.choice(GREETS)

    # ── FAREWELL ──────────────────────────────────────────
    elif intent == "farewell":
        cart = session.get("cart", [])
        if cart:
            total = sum(i["price"]*i["qty"] for i in cart)
            text = f"Goodbye! 👋 You have {len(cart)} item(s) in your cart — ${total:.2f} total. Come back to complete your order!"
        else:
            text = "Goodbye! 👋 Thanks for visiting. Come back anytime!"

    # ── HELP ──────────────────────────────────────────────
    elif intent == "help":
        text = ("🤖 **Here's what I can do:**\n\n"
                "🔍 Search — 'show me branded socks'\n"
                "🖼️ Images — every result shows the real product photo\n"
                "🛒 Buy — direct Amazon links on every card\n"
                "💰 Price — 'how much is this TV'\n"
                "📦 Stock — 'is this in stock'\n"
                "🗺️ Locate — 'where are the TVs in the store'\n"
                "⚖️ Compare — 'compare Nike vs Adidas socks'\n"
                "⭐ Recommend — 'best charger under $20'\n"
                "🏷️ Deals — 'any discounts on electronics'\n\n"
                "Just type naturally — I understand everyday language!")

    # ── FIND LOCATION ─────────────────────────────────────
    elif intent == "find_location":
        kw = " ".join(kws[:2]) if kws else query
        loc = get_location(kw)
        if not loc:
            # Try finding from recent products
            last = session.get("last_products", [])
            if last:
                loc = get_location(last[0].get("category",""))
        if loc:
            location = loc
            arrows = {"North Wing":"⬆️","South Wing":"⬇️","East Hall":"➡️","West Hall":"⬅️","Central Area":"🎯"}
            arrow  = arrows.get(loc["section"],"📍")
            text = (f"🗺️ Found it! Here's exactly where to go:\n\n"
                    f"📦 **{loc['category']}**\n\n"
                    f"🏢 **Floor:** {loc['floor']}\n"
                    f"{arrow} **Section:** {loc['section']}\n"
                    f"🛒 **Aisle:** {loc['aisle']}\n"
                    f"📚 **Shelf:** {loc['shelf']}\n\n"
                    f"💡 *Look for the '{loc['aisle']}' sign in the {loc['section']}.*")
            # Also show products from that category
            products = semantic_search(kw, top_k=4)
        else:
            text = ("🗺️ Tell me what product or category you're looking for and I'll guide you! "
                    "E.g. 'where are the TVs?' or 'which aisle is electronics?'")

    # ── PRICE CHECK ───────────────────────────────────────
    elif intent == "check_price":
        products = semantic_search(query, top_k=4, price_range=price_range)
        if products:
            p = products[0]
            disc = f" 🎉 **{p['discount']}% off** the original ${p['list_price']:.2f}!" if p["discount"] > 0 else ""
            text = (f"💰 **{p['title'][:70]}**\n\n"
                    f"Price: **${p['price']:.2f}**{disc}\n"
                    f"⭐ {p['stars']}/5 from {p['reviews']:,} reviews\n\n"
                    f"Here are similar options to compare:")
        else:
            text = "💭 I couldn't find that product. Try a different name or category!"

    # ── STOCK CHECK ───────────────────────────────────────
    elif intent == "check_stock":
        products = semantic_search(query, top_k=3, price_range=price_range)
        if products:
            p = products[0]
            if p["is_best_seller"]:
                status = "⚠️ **Best Seller — moving fast!** Limited stock, order soon."
            elif p["bought"] > 5000:
                status = "✅ **In Stock** — High demand. Ships today!"
            elif p["bought"] > 0:
                status = "✅ **In Stock** — Available now."
            else:
                status = "📋 **Check in-store** — Visit the aisle or ask a staff member."
            text = f"{status}\n\n**{p['title'][:70]}** — ${p['price']:.2f}"
        else:
            text = "📦 Which product would you like to check availability for?"

    # ── COMPARE ───────────────────────────────────────────
    elif intent == "compare":
        products = semantic_search(query, top_k=2, price_range=price_range)
        if len(products) >= 2:
            p1, p2 = products[0], products[1]
            cheaper  = p1["title"][:30] if p1["price"] <= p2["price"] else p2["title"][:30]
            higher_r = p1["title"][:30] if p1["stars"] >= p2["stars"] else p2["title"][:30]
            text = (f"⚖️ **Comparison**\n\n"
                    f"| | Product 1 | Product 2 |\n"
                    f"|---|---|---|\n"
                    f"| 💰 Price | ${p1['price']:.2f} | ${p2['price']:.2f} |\n"
                    f"| ⭐ Stars | {p1['stars']}/5 | {p2['stars']}/5 |\n"
                    f"| 📝 Reviews | {p1['reviews']:,} | {p2['reviews']:,} |\n"
                    f"| 🔥 Bought | {p1['bought']:,}/mo | {p2['bought']:,}/mo |\n\n"
                    f"💡 Better price: **{cheaper}**\n"
                    f"💡 Higher rated: **{higher_r}**")
        else:
            products = semantic_search(query, top_k=6)
            text = "⚖️ Here are similar products — click **Compare** on any two!"

    # ── RECOMMEND ─────────────────────────────────────────
    elif intent == "recommend":
        products = semantic_search(query, top_k=8, price_range=price_range)
        pr_str = f" under ${price_range['max']:.0f}" if price_range and "max" in price_range else ""
        cat_str = " ".join(kws[:2]).title() if kws else "products"
        text = (f"⭐ **Top picks for {cat_str}{pr_str}**\n\n"
                f"Ranked by rating, popularity, and relevance to your query. "
                f"Click any image to view on Amazon, or hit **Buy Now** to purchase directly!")

    # ── DISCOUNT ──────────────────────────────────────────
    elif intent == "discount":
        # Search for discounted items
        q = " ".join(kws) if kws else "popular products"
        candidates = semantic_search(q, top_k=20)
        discounted = [p for p in candidates if p["discount"] > 0]
        products   = (discounted or candidates)[:8]
        text = (f"🏷️ **Current Deals**\n\n"
                f"🔥 Up to 40% off selected items · Free shipping on orders over $35\n"
                f"Use code **SAVE15** for an extra 15% off!\n\n"
                f"Here are today's best discounted picks:")

    # ── ADD TO CART ───────────────────────────────────────
    elif intent == "add_to_cart":
        last = session.get("last_products", [])
        if last:
            p    = last[0]
            cart = session.get("cart", [])
            found = False
            for item in cart:
                if item["asin"] == p["asin"]:
                    item["qty"] += 1; found = True; break
            if not found:
                cart.append({"asin":p["asin"],"title":p["title"][:60],
                             "price":p["price"],"img":p.get("img_url",""),"qty":1})
            session["cart"] = cart
            total = sum(i["price"]*i["qty"] for i in cart)
            text = (f"🛒 Added **{p['title'][:55]}** to your cart!\n\n"
                    f"💰 Cart total: **${total:.2f}** ({len(cart)} item{'s' if len(cart)>1 else ''})")
            products = [p]
        else:
            products = semantic_search(query, top_k=4)
            text = "🛒 Which product would you like to add? Click **Add to Cart** on any card below!"

    # ── REVIEWS ───────────────────────────────────────────
    elif intent == "check_reviews":
        products = semantic_search(query, top_k=4)
        if products:
            p = products[0]
            bar = "⭐" * round(p["stars"]) + "☆" * (5 - round(p["stars"]))
            text = (f"📊 **{p['title'][:70]}**\n\n"
                    f"{bar} **{p['stars']}/5** from {p['reviews']:,} reviews\n"
                    f"🔥 {p['bought']:,} bought last month\n\n"
                    f"Customers love the quality and value. Click **View on Amazon** for the full review list!")
        else:
            text = "📝 Which product would you like reviews for?"

    # ── RETURN POLICY ─────────────────────────────────────
    elif intent == "return_policy":
        text = ("↩️ **Return & Refund Policy**\n\n"
                "✅ 30-day returns on most items\n"
                "✅ Free return shipping on defective items\n"
                "✅ Full refund within 5–7 business days\n"
                "✅ Size/color exchanges for clothing & shoes\n"
                "⚠️ Electronics: unopened only for full refund\n\n"
                "Start a return with your Order ID in the help menu!")

    # ── TRACK ORDER ───────────────────────────────────────
    elif intent == "track_order":
        text = ("📦 **Order Tracking**\n\n"
                "To track your order, please provide your **Order ID** "
                "and I'll fetch the latest status. Alternatively, visit "
                "your account's Order History for real-time updates.\n\n"
                "Typical delivery: 2–5 business days after dispatch.")

    # ── COMPLAINT ─────────────────────────────────────────
    elif intent == "complaint":
        text = ("😔 I'm really sorry to hear that. Your experience matters to us and "
                "I'm flagging this for our support team right now. "
                "Could you share the product name or order ID so we can resolve this for you?")
        escalate = True

    # ── DEFAULT: SEMANTIC SEARCH ──────────────────────────
    else:
        products = semantic_search(query, top_k=10, price_range=price_range)
        if products:
            pr_str = f" under ${price_range['max']:.0f}" if price_range and "max" in price_range else ""
            text = f"🛍️ Found **{len(products)} products**{pr_str} matching your search. Each card shows the live image and a direct Amazon buy link!"
        else:
            text = "🔍 I couldn't find exact matches. Try different keywords or browse a category!"
        response_type = "search"

    # Update session context
    if products:
        session["last_products"] = products[:3]
    session["last_intent"] = intent

    return {
        "text":      text,
        "products":  products,
        "location":  location,
        "type":      response_type,
        "escalate":  escalate,
        "cart":      session.get("cart", []),
    }


# ══════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    user_msg   = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Session
    if session_id not in sessions:
        sessions[session_id] = {"cart": [], "last_products": [], "last_intent": None, "turn": 0}
    session = sessions[session_id]
    session["turn"] += 1

    # NLP
    intent  = classify_intent(user_msg)
    sent    = sentiment(user_msg)
    t0      = time.time()
    resp    = build_response(intent, user_msg, session)
    ms      = round((time.time() - t0) * 1000)

    # Persist conversation
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO conversations (session_id,user_msg,bot_msg,intent,sentiment) VALUES (?,?,?,?,?)",
                 (session_id, user_msg, resp["text"][:500], intent, sent["label"]))
    conn.commit(); conn.close()

    return jsonify({
        "session_id": session_id,
        "response":   resp["text"],
        "products":   resp["products"],
        "location":   resp["location"],
        "type":       resp["type"],
        "escalate":   resp["escalate"],
        "cart":       resp["cart"],
        "nlp": {
            "intent":     intent,
            "sentiment":  sent,
            "keywords":   extract_keywords(user_msg),
            "price_range":extract_price(user_msg),
            "ms":         ms,
        }
    })

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    q    = data.get("query","")
    pr   = extract_price(q)
    results = semantic_search(q, top_k=data.get("top_k",10), price_range=pr)
    return jsonify({"results": results, "count": len(results)})

@app.route("/api/location", methods=["GET"])
def api_location():
    cat = request.args.get("category","")
    loc = get_location(cat)
    return jsonify(loc or {"error": "Not found"})

@app.route("/api/categories", methods=["GET"])
def api_categories():
    cats = df["category_name"].value_counts().head(40).reset_index()
    cats.columns = ["name","count"]
    return jsonify(cats.to_dict(orient="records"))

@app.route("/api/session/<sid>", methods=["GET"])
def api_session(sid):
    s = sessions.get(sid, {})
    return jsonify({"cart": s.get("cart",[]), "turn": s.get("turn",0),
                    "last_intent": s.get("last_intent")})

@app.route("/api/stats", methods=["GET"])
def api_stats():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT intent, COUNT(*) as n FROM conversations GROUP BY intent ORDER BY n DESC")
    intents = [{"intent":r[0],"count":r[1]} for r in c.fetchall()]
    c.execute("SELECT sentiment, COUNT(*) as n FROM conversations GROUP BY sentiment")
    sentiments = [{"label":r[0],"count":r[1]} for r in c.fetchall()]
    conn.close()
    return jsonify({"intents": intents, "sentiments": sentiments,
                    "total_products": len(df), "sessions": len(sessions)})

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_system()
    print("🌐 Server: http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
