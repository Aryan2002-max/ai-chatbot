

import os, json, csv, uuid, re
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


import os
import time

last_modified_time = 0
last_checked_time = 0
CHECK_INTERVAL = 60   # check every 60 seconds

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def check_and_reload_pdf():
    global last_modified_time, last_checked_time
    
    current_time = time.time()
    
    if current_time - last_checked_time < CHECK_INTERVAL:
        return
    
    last_checked_time = current_time
    
    pdf_path = "data/account.pdf"
    
    if not os.path.exists(pdf_path):
        print("PDF not found!")
        return
    
    current_modified_time = os.path.getmtime(pdf_path)
    
    if current_modified_time != last_modified_time:
        print("PDF updated! Reloading...")
        
        # 👉 YOUR PDF + FAISS CODE HERE
        
        last_modified_time = current_modified_time


import re

def is_valid_name(text):
    if len(text) < 3:
        return False
    return bool(re.match("^[A-Za-z ]+$", text))

def is_valid_address(text):
    return len(text) > 5

def is_valid_phone(text):
    return text.isdigit() and len(text) == 10

def is_valid_email(text):
    return "@" in text and "." in text




product_examples = [


    "wrong item", "wrong order", "galat item", "galat order",
    "item missing", "missing item", "item nahi mila",
    "kuch item missing", "order incomplete",


    "bad food", "food quality", "khana kharab", "food kharab",
    "food stale", "basi khana", "taste kharab", "khana bekar",
    "food spoiled", "khana kharab ho gaya",


    "damaged item", "item damaged", "item toot gaya",
    "product damaged", "product toot gaya",
    "package damaged", "parcel damaged",


    "cold food", "thanda khana", "food cold",
    "khana thanda tha",


    "expired product", "expiry cross", "expired item",
    "product kharab", "product kharab nikla",


    "wrong product", "size different", "different size",
    "galat size", "size galat", "different product",


    "less quantity", "kam quantity", "quantity kam",
    "food kam mila",


    "bad packaging", "packing kharab",
    "package open tha", "seal broken",

    ]

delivery_examples = [


    "delivery late", "late delivery",
    "delivery bahut late", "delivery me delay",
    "delivery delay", "bahut late deliver",
    "bahut der se aya", "late aya",


    "delivery boy rude", "delivery rude",
    "rider rude", "delivery boy misbehaved",
    "delivery boy bad behaviour",
    "delivery boy scolded", "delivery boy gussa",
    "delivery boy badtameezi", "delivery boy battameez",


    "delivery boy call nahi kiya",
    "rider call nahi kiya",
    "call nahi aya delivery boy ka",


    "fake delivery", "delivered but not received",
    "delivered dikha raha", "order received nahi hua",


    "wrong address delivery", "galat jagah deliver",
    "kisi aur ko deliver kar diya",


    "delivery boy call nahi kiya",
    "rider call nahi kiya",
    "call nahi aya delivery boy ka",


    "extra money", "extra charge",
    "paise maang raha", "extra paise",
    "delivery boy paise maang raha"]

general_examples = [
    "payment issue",
    "refund nahi mila",
    "login problem",
    "app crash",
    "account issue",
    "coupon not working"]
order_examples = [
    "order late",
    "order not delivered",
    "order kaha hai",
    "order track",
    "order delay ho gaya",
    "order missing",
    "cancel my order",
    "where is my order",
    "track my order",
    "order status","order cancel karo","please cancel my order","i do not want this order so cancel it"
]

complaint_texts = delivery_examples + product_examples + general_examples + order_examples


complaint_labels = (
    ["DELIVERY"] * len(delivery_examples) +
    ["PRODUCT"] * len(product_examples) +
    ["ORDER"] * len(order_examples) +
    ["GENERAL"] * len(general_examples)
)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

complaint_embeddings = embed_model.encode(
    complaint_texts,
    normalize_embeddings=True
)

complaint_index = faiss.IndexFlatIP(complaint_embeddings.shape[1])
complaint_index.add(np.array(complaint_embeddings))



def detect_complaint_type(text):
    text = text.lower()


    general_keywords = [
        "refund","payment","login","account","coupon",
        "app","crash","upi","money","deducted"
    ]

    order_keywords = [
        "order","cancel","track","status","where is my order"
    ]



    if any(k in text for k in order_keywords):
        return "ORDER"

    if any(k in text for k in general_keywords):
        return "GENERAL"


    emb = embed_model.encode([text], normalize_embeddings=True)
    D, I = complaint_index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])
    pred = complaint_labels[idx]


    if score < 0.35:
        return "GENERAL"

    return pred

MEMORY_FILE="chat_history.json"

def load_history():
    if not os.path.exists(MEMORY_FILE): return {}
    return json.load(open(MEMORY_FILE))

def save_history(h): json.dump(h,open(MEMORY_FILE,"w"),indent=4)

def store_chat(uid,u,b):
    h=load_history(); h.setdefault(uid,[]).append({"u":u,"b":b}); save_history(h)

def get_last_user_messages(uid,n=2):
    h=load_history()
    if uid not in h: return ""
    return " ".join([c["u"] for c in h[uid][-n:]])

def load_pdf(path):
    text=""
    for p in PdfReader(path).pages: text+=p.extract_text()+"\n"
    chunks=text.split("Q:")
    qs,ans=[],[]
    for c in chunks[1:]:
        p=c.split("A:")
        if len(p)==2:
            qs.append(p[0].strip()); ans.append(p[1].strip())
    return qs,ans

questions,answers=load_pdf("data/account.pdf")
emb=embed_model.encode(questions,normalize_embeddings=True)
index=faiss.IndexFlatIP(emb.shape[1]); index.add(np.array(emb))

def ask_pdf(q):
    e=embed_model.encode([q],normalize_embeddings=True)
    D,I=index.search(e,1)
    return answers[I[0][0]] if float(D[0][0])>0.6 else "NOT_FOUND"

intent_examples = {


 "SYSTEM": {
        "GREETING":[
            "hi","hello","hey","good morning","good evening","namaste"
        ],
        "THANKS":[
            "thanks","thank you","thanks a lot","thx","thnku","thnks","thankyou","thankyou so much"
        ],
        "BYE":[
            "bye","goodbye","see you later"
        ],
        "IDENTITY":[
            "who are you",
            "what can you do",
            "are you human",
            "what services you provide",
            "tum kon ho"
        ]
    },

"SUPPORT_QUERY":[ "refund nahi mila",
"refund delay","app crash","app is having error","downtime a rha he",
"refund not received","refund stuck",
"mera refund kaha hai","refund kitne din me aata",
"refund abhi tak nahi aya","paise wapas nahi mile","account issue","coupon not working","coupon kam nhi krrha"


"payment failed","payment deducted",
"money deducted but order not placed",
"double payment","upi payment issue",
"payment problem","paise kat gaye",
"payment ho gaya order nahi aya",


"cancel my order",
"order not delivered","order late",
"order kaha hai","order track",
"order delay ho gaya","order missing",


"delivery boy rude","rider rude",
"delivery boy misbehave","delivery boy late",
"delivery boy problem","delivery boy behaviour",
"rider bad behaviour","delivery late aya",


"wrong item delivered","wrong order",
"bad food","food quality bad","thanda khana",
"food kharab","food stale","food spoiled","package is open","parcel is open",
"stale food","cold food","packing is damage","packing is open","parcel is damaged",
"item missing","damaged product","seal broken",
"product broken","food spoiled",
"taste bad","restaurant problem",


"support reply nahi kar raha",
"no response from support",
"bahut din ho gaye","no update",
"ab kya karu","help me",
"complaint karni hai","register complaint",
"customer care se baat karni hai",

],
 "BUSINESS_QUERY":["REFUND KAB ATA HE","WHEN I RECEIVE MY REFUND"]

}

intent_examples["BUSINESS_QUERY"].extend(questions)

txt, lab = [], []

for intent, data in intent_examples.items():

    if intent == "SYSTEM":
        for sub_intent, examples in data.items():
            for ex in examples:
                txt.append(ex)
                lab.append(intent)


    else:
        for ex in data:
            txt.append(ex)
            lab.append(intent)

intent_embeddings = embed_model.encode(txt, normalize_embeddings=True)
intent_index = faiss.IndexFlatIP(intent_embeddings.shape[1])
intent_index.add(intent_embeddings)

sys_txt, sys_lab = [], []

for sub_intent, examples in intent_examples["SYSTEM"].items():
    for ex in examples:
        sys_txt.append(ex)
        sys_lab.append(sub_intent)

sys_emb = embed_model.encode(sys_txt, normalize_embeddings=True)
system_index = faiss.IndexFlatIP(sys_emb.shape[1])
system_index.add(sys_emb)

def detect_system_subintent(text):

    emb = embed_model.encode([text], normalize_embeddings=True)
    D, I = system_index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])
    label = sys_lab[idx]

    if score < 0.75:
        return None

    return label

def detect_intent(text):

    text = text.lower().strip()

    sys_sub = detect_system_subintent(text)
    if sys_sub in ["GREETING","THANKS","BYE","IDENTITY"]:
        return "SYSTEM"

    emb = embed_model.encode([text], normalize_embeddings=True)
    D, I = intent_index.search(emb, 1)

    score = float(D[0][0])
    idx = int(I[0][0])
    intent = lab[idx]

    if score < 0.40:
        return "NOISE"

    return intent

def system_reply(msg):

    sub = detect_system_subintent(msg)

    if sub == "GREETING":
        return "Hello , I am your AI customer support assistant. How can I help you?"

    if sub == "THANKS":
        return "You're welcome , Happy to help!"

    if sub == "BYE":
        return "Goodbye , Have a nice day!"

    if sub == "IDENTITY":
        return (
            "I am an AI Customer Support Assistant .\n"
            "I can help you with:\n"
            "• Order issues\n"
            "• Delivery complaints\n"
            "• Product complaints\n"
            "• General support questions"
        )

    return "How can I assist you?"

TICKET_FILE="tickets.csv"
def ticket_id(): return str(uuid.uuid4())[:8].upper()

def create_ticket(d):
    id=ticket_id(); f=os.path.exists(TICKET_FILE)
    with open(TICKET_FILE,"a",newline="") as x:
        w=csv.writer(x)
        if not f:
            w.writerow(["id","name","phone","email","address",
            "order_id",
            "delivery_boy","delivery_phone","restaurant",
            "details","status"])
        w.writerow([id,d.get("name"),d.get("phone"),d.get("email"),
            d.get("address"),
            d.get("order_id",""),
            d.get("delivery_name",""),
            d.get("delivery_phone",""),
            d.get("restaurant",""),
            d.get("details"),"OPEN"])
    return id


import uuid
import random


user_states = {}


import random

user_states = {}

import re
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def handle_support_flow(uid, msg):
    msg = msg.strip()


    if uid not in user_states:
        ctype = detect_complaint_type(msg)

        user_states[uid] = {
            "step": "name",
            "type": ctype,
            "complaint": msg
        }

        return "Please share your name."

    state = user_states[uid]
    step = state["step"]
    ctype = state["ctype"]

    if step == "name":
        state["name"] = msg
        state["step"] = "phone"
        return "Please share your phone number."


    if step == "phone":
        if not msg.isdigit() or len(msg) != 10:
            return "Please enter valid 10 digit phone number."

        state["phone"] = msg
        state["step"] = "email"
        return "Please share your email."


    if step == "email":
        if not is_valid_email(msg):
            return "Please enter valid email."

        state["email"] = msg
        state["step"] = "address"
        return "Please share your address."


    if step == "address":
        state["address"] = msg

        ctype = state.get("ctype")


        if ctype == "DELIVERY":
            state["step"] = "delivery_name"
            return "Please share delivery boy name."


        elif ctype == "PRODUCT":
            state["step"] = "restaurant"
            return "Please share restaurant/shop name."


        elif ctype == "ORDER":
            state["step"] = "order_id"
            return "Please share your Order ID."


        else:
             state["details"] = state.get("complaint",msg)
             ticket = create_ticket(state)
             user_states.pop(uid)
             return f"Complaint registered successfully! Your ticket id is {ticket}"

    if step == "delivery_name":
        state["delivery_name"] = msg
        state["step"] = "delivery_phone"
        return "Please share delivery boy phone number."

    if step == "delivery_phone":
        if not msg.isdigit() or len(msg) != 10:
            return "Please enter valid delivery boy phone."

        state["delivery_phone"] = msg
        state["details"] = state.get("complaint")
        ticket = create_ticket(state)
        user_states.pop(uid)
        return f"Complaint registered successfully! Your ticket id is {ticket}"

    if step == "restaurant":
        state["restaurant"] = msg
        state["details"] = state.get("complaint")
        ticket = create_ticket(state)
        user_states.pop(uid)
        return f"Complaint registered successfully! Your ticket id is {ticket}"


    if step == "order_id":
        state["order_id"] = msg
        state["details"] = state.get("complaint",msg)

        ticket = create_ticket(state)
        user_states.pop(uid)

        return f"Complaint registered successfully! Your ticket id is {ticket}"

def chatbot(uid, msg):

    check_and_reload_pdf()
    try:
        # 🔁 If user already in flow
        if uid in user_states:
            return handle_support_flow(uid, msg)

        # ⚙️ System intent
        sys_type = detect_system_subintent(msg)
        if sys_type:
            return system_reply(msg)

        # 🎯 Main intent detection
        intent = detect_intent(msg)

        # 📊 Business Query (PDF / FAISS)
        if intent == "BUSINESS_QUERY":
            pdf = ask_pdf(msg)
            if pdf != "NOT_FOUND":
                return pdf
            else:
                return "Iska answer abhi available nahi hai."

        # ⚙️ System queries
        if intent == "SYSTEM":
            return system_reply(msg)

        # 🛠️ Support query
        if intent == "SUPPORT_QUERY":
            ctype = detect_complaint_type(msg)

            user_states[uid] = {
                "step": "name",
                "ctype": ctype,
                "complaint": msg
            }

            return "I understand your issue. Please share your full name."

        # 🔇 Noise
        if intent == "NOISE":
            return "Sorry, I didn't understand. Please rephrase."

        # 🧠 Default fallback
        return "How can I assist you?"

    except Exception as e:
        return f"❌ Error: {str(e)}"

