import re
import unicodedata

_MAP=str.maketrans({"ø":"o","Ø":"O","đ":"d","Đ":"D","ł":"l","Ł":"L","ß":"ss",
                    "æ":"ae","Æ":"AE","œ":"oe","Œ":"OE","ð":"d","Ð":"D","þ":"th","Þ":"Th","ı":"i","İ":"I","ş":"s","Ş":"S","ğ":"g","Ğ":"G"})
def norm(s):
    if not isinstance(s,str): return ""
    s=s.translate(_MAP)
    s=s.replace("'","").replace("\u2019","").replace("-"," ")
    s=unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode()
    return re.sub(r"\\s+"," ",re.sub(r"[^a-z ]"," ",s.lower())).strip()

def parse_listone(nome):
    """'Martinez L.'->('martinez',['l']); 'Martinez Jo.'->('martinez',['jo']);
       'Esposito F.P.'->('esposito',['f','p'])"""
    n=nome.strip()
    m=re.match(r"^(.*?)\s+((?:[A-Z][a-z]{0,3}\.\s*)+)$", n)
    if m:
        sur=norm(m.group(1))
        pre=[x.lower() for x in re.findall(r"([A-Z][a-z]{0,3})\.", m.group(2))]
    else:
        sur=norm(n); pre=[]
    return sur, pre

def build_index(players, teams):
    rows=[]
    for p,t in zip(players,teams):
        np_=norm(p); toks=np_.split()
        if not toks: continue
        rows.append({"full":p,"nfull":np_,"toks":toks,"team":t})
    return rows
def match_one(sur,ini,team,index):
    st=sur.split()
    cands=[]
    for r in index:
        tk=r["toks"]
        # surname tokens must appear as a contiguous suffix-ish block
        ok=False
        for i in range(len(tk)-len(st)+1):
            if tk[i:i+len(st)]==st: ok=True; break
        if not ok: continue
        score=0.0
        # initials of the preceding given names
        pre=[x for x in tk if x not in st]
        if ini:
            hits=sum(1 for c in ini if any(g.startswith(c) for g in pre))
            if hits==len(ini):                                  score+=2.5
            elif hits>0:                                        score+=1.0
            elif pre and any(g[0]==ini[0][0] for g in pre):     score+=0.3
            else:                                               score-=1.5
        else:
            if not pre: score+=1.0
            score+=0.3
        if team and r["team"]==team: score+=2.5
        score+= 0.5/(1+abs(len(tk)-len(st)-len(ini)))
        cands.append((score,r))
    if not cands: return None,0.0
    cands.sort(key=lambda x:-x[0])
    if cands[0][0] <= 0.0: return None, cands[0][0]
    return cands[0][1], cands[0][0]
