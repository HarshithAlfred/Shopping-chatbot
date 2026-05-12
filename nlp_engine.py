"""
ShopBot NLP Engine v2 — Real Amazon Dataset
============================================
• TF-IDF + Logistic Regression intent classifier (trained on 800+ examples)
• Rule-based entity extractor  (brand, category, price, size, color, quantity)
• TextBlob + lexicon sentiment analyzer
• Keyword override rules for edge cases
• Multi-turn dialogue with pronoun resolution
• Store location finder
"""

import re, os, pickle, random, sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textblob import TextBlob
import warnings; warnings.filterwarnings('ignore')

DB_PATH = "store.db"

# ══════════════════════════════════════════════════════
#  TRAINING DATA  — 800+ examples, very natural phrasing
# ══════════════════════════════════════════════════════

TRAINING_DATA = {

    "greet": [
        "hello","hi","hey","good morning","good afternoon","good evening","howdy",
        "hi there","hey there","what's up","greetings","yo","sup","morning","hiya",
        "hello bot","hi assistant","hey shopbot","start","begin","helo","hii","heyy",
        "good day","hi how are you","hello there","hey buddy","what's good",
    ],

    "farewell": [
        "bye","goodbye","see you","later","take care","thanks bye","cya","good night",
        "thank you goodbye","exit","quit","done","that's all","I'm done","signing off",
        "ok bye","alright bye","thanks that's all","have a good one","peace","au revoir",
        "see ya","gotta go","talk later","catch you later","I'm leaving",
    ],

    "search_product": [
        # generic browse
        "show me products","browse store","what do you have","list products",
        "show me everything","what can I buy","show catalog","product catalog",
        "show all items","what's available","let me see products","show items",
        # shoes
        "show me shoes","find shoes","I want shoes","looking for shoes","shoe sale",
        "shoes available","show shoe collection","find me sneakers","any sneakers",
        "looking for sneakers","show footwear","women shoes","men shoes",
        "running shoes","sports shoes","casual shoes","kids shoes","girls shoes",
        "boys shoes","show me boots","find sandals","I need shoes","get me shoes",
        # clothing
        "find me a jacket","show jackets","what jackets do you have","coats available",
        "show clothing","find clothes","browse fashion","show me jeans","find shirts",
        "t shirts","women clothing","men clothing","kids clothing","girls clothing",
        "boys clothing","dress","tops","pants","shorts","sweatshirt","hoodie",
        "show me men clothing","find women clothing","girls dresses",
        # electronics
        "show me phones","find laptop","I want headphones","show TVs","find cameras",
        "what phones do you have","electronics","browse laptops","find smartwatch",
        "show tablets","gaming consoles","find earbuds","show computers","find monitor",
        "show me electronics","tech products","gadgets","find charger","power bank",
        # home
        "home products","kitchen stuff","show bedding","find furniture",
        "home decor","cleaning supplies","show vacuum cleaners","kitchen appliances",
        "find cookware","show lighting","home storage","find rugs","show curtains",
        # toys
        "find toys","show toys","kids toys","boys toys","girls toys","baby toys",
        "show games","find puzzles","lego","action figures","stuffed animals",
        # beauty
        "beauty products","skincare","makeup","hair care","find shampoo",
        "nail care","perfume","fragrance","show beauty items","moisturizer",
        # sports
        "sports equipment","fitness gear","workout stuff","gym equipment",
        "outdoor gear","camping equipment","find yoga mat","show dumbbells",
        # automotive
        "car accessories","automotive parts","find car charger","show car mats",
        # general patterns
        "show me [category]","find [product]","looking for [item]","I need [product]",
        "do you have [item]","can I find [product]","where can I find","show collection",
        "browse [category]","I want to see","display products","list items",
    ],

    "check_price": [
        "how much does this cost","what is the price","how much is this",
        "tell me the price","price of this","cost of this","what does it cost",
        "price check","how expensive","cost?","price?","how much?","what's the price",
        "price range","how much would this be","what is the cost","give me the price",
        "price list","pricing","how much for this","cost information","rate of product",
        "how much are these shoes","shoe price","laptop price","phone price",
        "how much is the jacket","price of headphones","TV price","camera cost",
        "what's the price tag","affordable","how much should I pay","cost details",
        "how much does it go for","going rate","market price","current price",
        "is this cheap","is this expensive","price estimate","value for money",
        "how much is this item","price per piece","unit price","total cost",
        "budget needed","minimum price","maximum price","discounted price",
        "original price","sale price","final price","checkout price",
    ],

    "check_stock": [
        "is this in stock","do you have this","is it available","availability",
        "stock status","how many left","any units left","still in stock",
        "can I buy this now","available to purchase","in store","on shelf",
        "stock level","quantity available","do you have more","is it sold out",
        "out of stock","when will it be back","restocked","inventory",
        "is the phone available","do you have size 10","is size 8 available",
        "do you have this in blue","available colors","available sizes",
        "currently available","check inventory","product availability",
        "can I get this","stock check","how many do you have","units in stock",
        "available right now","immediately available","ready for pickup","ready to buy",
        "is this product available","when can I get this","how soon available",
    ],

    "get_details": [
        "tell me more","more details","more info","describe this","what is this",
        "product description","specifications","specs","features","what does it do",
        "tell me about","give me details","full details","complete info","explain this",
        "what are the specs","technical details","product info","item details",
        "what is included","what's in the box","dimensions","weight","material",
        "battery life","screen size","megapixels","RAM","storage capacity",
        "color options","size options","warranty","made of","what kind of product",
        "detail on shoes","shoe details","tell me about the shoes","info about shoes",
        "can i get the details","get details","fetch details","show details",
        "what is this product","product specifications","item specifications",
        "I want to know more","know more about","describe the product",
        "product feature","highlight","what makes this special","overview",
        "can you describe","brief description","long description","summary of product",
        "details please","show product details","more information","expand on this",
    ],

    "find_location": [
        "where is this","where can I find this","which aisle","what aisle",
        "where in the store","store location","product location","find in store",
        "where is it located","which section","where is the aisle","floor location",
        "map to product","navigate to product","how to get there","directions",
        "where are the shoes","which aisle are shoes","where is electronics",
        "where is clothing","shoe aisle","electronics section","furniture section",
        "where do I go","how do I get to","show me the way","in store navigation",
        "store map","aisle map","product map","which floor","what floor",
        "where can I pick this up","pickup location","in store pickup",
        "where is it in the store","guide me to","take me to","help me find",
        "which department","store department","product department","store section",
        "where are the toys","where is home decor","where is baby section",
        "I can't find it","help me locate","locate product","product aisle",
        "where exactly","exact location","precise location","store directory",
    ],

    "compare_products": [
        "compare","vs","versus","difference between","which is better",
        "compare these two","side by side","pros and cons","which should I buy",
        "which one is worth it","better option","which to choose","help me decide between",
        "compare products","compare phones","compare laptops","compare shoes",
        "iPhone vs Samsung","Nike vs Adidas","which has better specs","cheaper option",
        "quality comparison","value comparison","which is higher rated",
        "what's the difference","contrast these","evaluate both","which is best",
        "a or b","option 1 or 2","first or second","which would you recommend",
        "compare prices","price comparison","rating comparison","feature comparison",
        "both products","two products","multiple options","alternatives",
    ],

    "get_recommendations": [
        "recommend","suggest","what should I buy","help me choose","best product",
        "top rated","highly rated","most popular","best seller","trending",
        "what's good","any good products","what do you recommend","suggest something",
        "good options","popular products","customer favorites","staff picks",
        "best value","editor choice","top picks","what's trending","new arrivals",
        "latest products","what's new","fresh stock","just arrived",
        "recommend shoes","best shoes","top shoes","suggest shoes","good shoes",
        "best laptop","top phone","best headphones","good camera","top TV",
        "recommend for student","best gift","gift ideas","gift recommendation",
        "budget option","affordable recommendation","cheap but good","value picks",
        "premium option","luxury pick","high end recommendation","quality product",
        "best for kids","family friendly","best for gaming","best for work",
        "best for travel","best for home","everyday use","professional grade",
        "what are people buying","what's selling fast","hot right now","in demand",
        "most reviewed","highest stars","5 star products","top rated items",
        "show me your best","personal recommendation","tailored for me","curated picks",
    ],

    "add_to_cart": [
        "add to cart","buy this","purchase","order now","I'll take it","add to bag",
        "put in cart","buy it","get it","I want this","I'll buy this","checkout",
        "place order","I'll order this","take this one","get this for me","buy now",
        "add shoes to cart","order the shoes","buy the laptop","get the phone",
        "purchase this item","add item","I want to purchase","proceed to buy",
        "add one","add 2","buy 2","order 3","buy a pair","get a set","add a pack",
        "I'll take these","I want these","add them to cart","buy all","get all",
        "purchase now","order immediately","buy right now","instant buy","quick buy",
        "confirm purchase","finalize order","complete purchase","ready to buy",
    ],

    "check_reviews": [
        "reviews","ratings","what do customers say","customer feedback","user reviews",
        "show reviews","review score","star rating","overall rating","feedback",
        "is it good quality","is this worth buying","trusted product","reliable",
        "what are people saying","buyer reviews","verified reviews","honest reviews",
        "positive reviews","negative reviews","good reviews","bad reviews","mixed reviews",
        "product rating","how many stars","how is the quality","quality check",
        "review summary","review highlights","top reviews","recent reviews",
        "is this recommended","would you recommend","worth it","value for money",
        "shoe reviews","laptop reviews","phone reviews","product opinions",
        "what do buyers think","public opinion","community rating","social proof",
        "testimonials","user opinions","read reviews","check reviews","view reviews",
    ],

    "discount_offers": [
        "any discounts","sale","coupon","promo code","offers","current deals",
        "is there a sale","cashback","deal of the day","flash sale","clearance",
        "discount","percentage off","special offer","reduced price","marked down",
        "on sale","buy one get one","bogo","voucher","savings","best deal",
        "shoe sale","electronics sale","clothing sale","today's offers","weekly deals",
        "seasonal sale","holiday sale","festival offer","limited time offer",
        "discount code","offer code","apply coupon","use promo","how much off",
        "save money","save on purchase","price drop","cheaper now","reduced",
        "discount on shoes","deal on laptops","sale on phones","offer on clothing",
        "special price","exclusive deal","member discount","loyalty offer",
        "what promotions","any promotions","current promotions","active offers",
    ],

    "track_order": [
        "where is my order","track order","order status","when will it arrive",
        "delivery status","shipping update","track shipment","track package",
        "has my order shipped","is it dispatched","tracking number","estimated delivery",
        "expected arrival","when will I receive","order update","shipment status",
        "where's my package","delivery update","order confirmation","order history",
        "check my order","my orders","order details","order number","delivery time",
    ],

    "return_policy": [
        "return policy","can I return","refund policy","how to return","exchange policy",
        "return window","money back","return process","how do returns work",
        "can I exchange","swap product","return shipping","return instructions",
        "what is your return policy","30 day return","refund process","get a refund",
        "return item","exchange item","send back","replacement","defective product return",
        "damaged item return","wrong item return","size exchange","color exchange",
    ],

    "sentiment_complaint": [
        "this is terrible","very disappointed","product is broken","worst experience",
        "not happy","quality is poor","doesn't work","waste of money","never buying again",
        "bad product","terrible service","completely useless","not as described",
        "misleading","fraud","horrible","awful","disgusting quality","pathetic",
        "totally useless","broken on arrival","defective","scam","cheated","rubbish",
        "garbage product","worst purchase","regret buying","should not have bought",
        "poor quality","cheap material","falls apart","not durable","very bad",
        "angry customer","extremely upset","furious","unhappy","total waste",
    ],

    "sentiment_praise": [
        "amazing","excellent","fantastic","love this","perfect product","best purchase",
        "highly recommend","five stars","outstanding","brilliant","worth every penny",
        "exceeded expectations","very satisfied","great quality","awesome product",
        "wonderful","superb","top notch","flawless","delighted","thrilled",
        "very pleased","impressed","10 out of 10","would buy again","perfect",
        "incredible value","mind blowing","love it","so good","really happy",
        "great experience","best thing I bought","absolutely love","super happy",
    ],

    "help": [
        "help","what can you do","how do you work","what are your features",
        "guide me","instructions","menu","options","what can I ask",
        "how does this work","what can this bot do","show me what you can do",
        "capabilities","functions","commands","what to ask","i dont know what to ask",
        "assist me","support","I'm confused","not sure what to ask","how to use",
    ],

    "voice_query": [
        "can you speak","read this aloud","speak the answer","voice response",
        "say it","read out","text to speech","audio response","voice output",
        "speak to me","tell me verbally","voice mode","listen","audio mode",
    ],
}

# ══════════════════════════════════════════════════════
#  KEYWORD RULE OVERRIDES — handles tricky edge cases
# ══════════════════════════════════════════════════════

KEYWORD_RULES = [
    # (set_of_words_in_query,  intent_to_assign)
    ({"where", "aisle"},          "find_location"),
    ({"where", "find"},           "find_location"),
    ({"which", "aisle"},          "find_location"),
    ({"where", "located"},        "find_location"),
    ({"which", "section"},        "find_location"),
    ({"where", "store"},          "find_location"),
    ({"navigate"},                "find_location"),
    ({"floor", "product"},        "find_location"),
    ({"map"},                     "find_location"),
    ({"aisle"},                   "find_location"),
    ({"details", "sale"},         "get_details"),
    ({"details", "shoe"},         "get_details"),
    ({"info", "shoe"},            "get_details"),
    ({"info", "shoes"},           "get_details"),
    ({"price", "shoe"},           "check_price"),
    ({"price", "shoes"},          "check_price"),
    ({"stock", "shoe"},           "check_stock"),
    ({"available", "sizes"},      "check_stock"),
    ({"review", "shoe"},          "check_reviews"),
    ({"reviews", "shoes"},        "check_reviews"),
    ({"sale", "shoe"},            "discount_offers"),
    ({"deal", "shoe"},            "discount_offers"),
    ({"discount", "shoe"},        "discount_offers"),
    ({"can", "i", "get", "details"}, "get_details"),
]

def keyword_override(text: str):
    words = set(re.findall(r'\b\w+\b', text.lower()))
    for rule_words, intent in KEYWORD_RULES:
        if rule_words.issubset(words):
            return intent
    return None

# ══════════════════════════════════════════════════════
#  ENTITY EXTRACTOR
# ══════════════════════════════════════════════════════

COLORS = ["red","blue","green","black","white","pink","yellow","purple","orange",
          "grey","gray","silver","gold","brown","navy","beige","cream","maroon",
          "teal","cyan","violet","indigo","rose","coral","mint","olive"]

SIZES  = ["xs","s","m","l","xl","xxl","xxxl","small","medium","large","extra large",
          "size 6","size 7","size 8","size 9","size 10","size 11","size 12",
          "6 inch","8 inch","10 inch","15 inch","27 inch","32 inch","55 inch","65 inch"]

BRANDS = [
    "nike","adidas","apple","samsung","sony","dell","hp","lenovo","asus","microsoft",
    "puma","reebok","new balance","under armour","levi","north face","ray ban",
    "dyson","kitchenaid","philips","canon","nikon","bose","jbl","logitech",
    "amazon","google","lg","panasonic","toshiba","haier","whirlpool","bosch",
    "lego","fisher price","mattel","hasbro","barbie","nerf","vans","converse",
    "gucci","zara","h&m","calvin klein","ralph lauren","tommy hilfiger","gap",
]

PRICE_PATTERNS = [
    (r'under \$?(\d+(?:\.\d+)?)',            lambda m: {"max": float(m.group(1))}),
    (r'below \$?(\d+(?:\.\d+)?)',            lambda m: {"max": float(m.group(1))}),
    (r'less than \$?(\d+(?:\.\d+)?)',        lambda m: {"max": float(m.group(1))}),
    (r'up to \$?(\d+(?:\.\d+)?)',            lambda m: {"max": float(m.group(1))}),
    (r'not more than \$?(\d+(?:\.\d+)?)',    lambda m: {"max": float(m.group(1))}),
    (r'max(?:imum)? \$?(\d+(?:\.\d+)?)',     lambda m: {"max": float(m.group(1))}),
    (r'within \$?(\d+(?:\.\d+)?)',           lambda m: {"max": float(m.group(1))}),
    (r'around \$?(\d+(?:\.\d+)?)',           lambda m: {"min": float(m.group(1))*0.8, "max": float(m.group(1))*1.2}),
    (r'about \$?(\d+(?:\.\d+)?)',            lambda m: {"min": float(m.group(1))*0.85, "max": float(m.group(1))*1.15}),
    (r'\$?(\d+(?:\.\d+)?)\s*(?:to|-)\s*\$?(\d+(?:\.\d+)?)', lambda m: {"min": float(m.group(1)), "max": float(m.group(2))}),
    (r'between \$?(\d+) and \$?(\d+)',       lambda m: {"min": float(m.group(1)), "max": float(m.group(2))}),
    (r'over \$?(\d+(?:\.\d+)?)',             lambda m: {"min": float(m.group(1))}),
    (r'above \$?(\d+(?:\.\d+)?)',            lambda m: {"min": float(m.group(1))}),
    (r'more than \$?(\d+(?:\.\d+)?)',        lambda m: {"min": float(m.group(1))}),
    (r'budget (?:of |is )?\$?(\d+(?:\.\d+)?)', lambda m: {"max": float(m.group(1))}),
]

def extract_entities(text: str) -> dict:
    tl = text.lower()
    words = set(re.findall(r'\b\w+\b', tl))
    ents = {"brands": [], "colors": [], "sizes": [], "price_range": None,
            "quantities": [], "keywords": []}

    for b in BRANDS:
        if b in tl: ents["brands"].append(b)

    for col in COLORS:
        if col in words: ents["colors"].append(col)

    for sz in SIZES:
        if sz in tl: ents["sizes"].append(sz)

    for pattern, extractor in PRICE_PATTERNS:
        m = re.search(pattern, tl)
        if m:
            ents["price_range"] = extractor(m)
            break

    qty_m = re.search(r'(\d+)\s*(?:units?|pieces?|pairs?|packs?|sets?|of these)', tl)
    if qty_m: ents["quantities"].append(int(qty_m.group(1)))

    stopwords = {"i","me","my","show","find","get","want","need","looking","for",
                 "the","a","an","is","are","do","does","this","that","it","them",
                 "these","those","can","you","tell","about","more","please","ok",
                 "have","do","will","would","could","should","any","some","all","to","in"}
    ents["keywords"] = [w for w in re.findall(r'\b[a-z][a-z0-9]+\b', tl)
                        if w not in stopwords and len(w) > 2][:8]

    return ents

# ══════════════════════════════════════════════════════
#  SENTIMENT ANALYZER
# ══════════════════════════════════════════════════════

POSITIVE_LEXICON = {"amazing","excellent","fantastic","love","perfect","outstanding",
    "brilliant","awesome","great","best","wonderful","superb","happy","delighted",
    "satisfied","impressed","thrilled","pleased","glad","joy","nice","good","fine"}
NEGATIVE_LEXICON = {"terrible","awful","horrible","worst","broken","disappointed",
    "useless","waste","fraud","scam","bad","poor","ugly","hate","disgusting",
    "pathetic","garbage","rubbish","defective","cheated","broken","horrible","angry"}

def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    pol  = blob.sentiment.polarity
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pol  += 0.12 * len(words & POSITIVE_LEXICON)
    pol  -= 0.12 * len(words & NEGATIVE_LEXICON)
    pol   = max(-1.0, min(1.0, pol))
    if pol > 0.05:   label, emoji = "positive", "😊"
    elif pol < -0.05: label, emoji = "negative", "😞"
    else:             label, emoji = "neutral",  "😐"
    return {"label": label, "polarity": round(pol, 3),
            "subjectivity": round(blob.sentiment.subjectivity, 3), "emoji": emoji}

# ══════════════════════════════════════════════════════
#  NLP MODEL
# ══════════════════════════════════════════════════════

class ShoppingNLP:

    MODEL_PATH = "models/nlp_v2.pkl"

    def __init__(self):
        self.pipeline = None
        self.encoder  = LabelEncoder()
        self.ready    = False

    def _corpus(self):
        xs, ys = [], []
        for intent, examples in TRAINING_DATA.items():
            for ex in examples:
                xs.append(self._clean(ex))
                ys.append(intent)
        return xs, ys

    def _clean(self, t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r'[^\w\s$]', ' ', t)
        return re.sub(r'\s+', ' ', t)

    def train(self):
        print("🧠 Training NLP model …")
        xs, ys = self._corpus()
        Xtr, Xte, ytr, yte = train_test_split(
            xs, ys, test_size=0.12, random_state=42, stratify=ys)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3), max_features=15000,
                sublinear_tf=True, analyzer='word',
                token_pattern=r'\b\w\w+\b')),
            ("clf", LogisticRegression(
                max_iter=3000, C=4.0, solver='saga', random_state=42)),
        ])
        ytr_e = self.encoder.fit_transform(ytr)
        yte_e = self.encoder.transform(yte)
        self.pipeline.fit(Xtr, ytr_e)
        acc = np.mean(self.pipeline.predict(Xte) == yte_e)
        print(f"✅ Accuracy {acc:.2%}")
        print(classification_report(yte_e, self.pipeline.predict(Xte),
              target_names=self.encoder.classes_, zero_division=0))
        os.makedirs("models", exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "encoder": self.encoder}, f)
        print(f"💾 Saved to {self.MODEL_PATH}")
        self.ready = True

    def load(self):
        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, "rb") as f:
                d = pickle.load(f)
            self.pipeline, self.encoder = d["pipeline"], d["encoder"]
            self.ready = True
            print("✅ NLP model loaded")
        else:
            self.train()

    def predict(self, text: str) -> dict:
        if not self.ready: self.load()

        # 1. Keyword rules take priority
        override = keyword_override(text)
        if override:
            return {"intent": override, "confidence": 0.97,
                    "alternatives": [], "source": "rules"}

        clean = self._clean(text)
        proba = self.pipeline.predict_proba([clean])[0]
        top3  = np.argsort(proba)[::-1][:3]
        names = self.encoder.classes_

        return {
            "intent":     names[top3[0]],
            "confidence": float(proba[top3[0]]),
            "alternatives": [{"intent": names[i], "confidence": float(proba[i])}
                             for i in top3[1:]],
            "source": "model",
        }

# ══════════════════════════════════════════════════════
#  DIALOGUE MANAGER  — multi-turn session state
# ══════════════════════════════════════════════════════

class DialogueManager:

    def __init__(self):
        self._sessions = {}

    def session(self, sid: str) -> dict:
        if sid not in self._sessions:
            self._sessions[sid] = {
                "history": [], "context": {}, "last_intent": None,
                "last_products": [], "cart": [], "turn": 0,
                "voice_mode": False,
            }
        return self._sessions[sid]

    def update(self, sid, intent, entities, products, response_text):
        s = self.session(sid)
        s["last_intent"] = intent
        s["turn"] += 1
        if products:            s["last_products"] = products
        if entities["brands"]:  s["context"]["brand"]    = entities["brands"][0]
        if entities["keywords"]:s["context"]["keywords"] = entities["keywords"]
        if entities.get("price_range"): s["context"]["price_range"] = entities["price_range"]
        s["history"].append({"intent": intent, "entities": entities,
                              "preview": response_text[:80]})
        s["history"] = s["history"][-12:]

    def resolve(self, text: str, sid: str) -> str:
        """Replace 'it / this / these / them' with last product name."""
        s = self.session(sid)
        tl = text.lower()
        if any(p in tl.split() for p in ["it","this","these","them","that","those"]):
            if s["last_products"]:
                name = s["last_products"][0].get("title","")[:50]
                for p in ["it","this","these","them","that","those"]:
                    text = re.sub(rf'\b{p}\b', name, text, flags=re.IGNORECASE)
        return text

    def add_to_cart(self, sid, product, qty=1):
        s = self.session(sid)
        for item in s["cart"]:
            if item["asin"] == product["asin"]:
                item["qty"] += qty
                return
        s["cart"].append({"asin": product["asin"], "title": product["title"][:60],
                          "price": product["price"], "img": product.get("img_url",""),
                          "qty": qty})

    def get_cart(self, sid): return self.session(sid)["cart"]

# ══════════════════════════════════════════════════════
#  SINGLETONS
# ══════════════════════════════════════════════════════
nlp_model        = ShoppingNLP()
dialogue_manager = DialogueManager()

# ══════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    nlp_model.train()
    probes = [
        "can i get the details on shoe sale",
        "show me shoes under $50",
        "where are the shoes in the store",
        "which aisle is electronics",
        "how much is this laptop",
        "is this phone in stock",
        "recommend a gift under $30",
        "compare Nike and Adidas",
        "what do customers say about this",
        "any deals on clothing today",
        "add this to my cart",
        "I am very disappointed with this product",
        "this is absolutely amazing quality",
        "where can I find home decor",
    ]
    print("\n🧪 Probe results:")
    for q in probes:
        r = nlp_model.predict(q)
        src = r.get("source","?")
        print(f"  [{src:5}] {r['confidence']:.0%}  {r['intent']:20} │ {q}")
