#!/usr/bin/env python3
"""Render the headline result as a bar chart. Numbers are taken verbatim from
the README result table (strategic 60-95% vs textual-criteria 3-7% instability;
binary relevance ~2% instability / 98% stable)."""
from PIL import Image, ImageDraw, ImageFont
import os
W,H=1120,560
BG=(255,255,255); INK=(23,30,40); MUT=(110,120,132); GRID=(228,232,238)
RED=(214,69,65); GRN=(40,160,90); BLUE=(70,120,210)
ARIALB="/System/Library/Fonts/Supplemental/Arial Bold.ttf"; ARIAL="/System/Library/Fonts/Supplemental/Arial.ttf"
def f(p,s):
    try: return ImageFont.truetype(p,s)
    except Exception: return ImageFont.load_default()
title=f(ARIALB,30); lab=f(ARIALB,21); sub=f(ARIAL,18); small=f(ARIAL,16); tick=f(ARIAL,15)
img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
d.text((50,30),"Classification instability by prompt framing",font=title,fill=INK)
d.text((50,72),"Same task, same documents, same model (Claude Sonnet 4) — only the prompt wording changes.",font=sub,fill=MUT)

# plot area
x0,x1=470,1060; ytop=150; rowh=120
def xof(p): return x0+(x1-x0)*p/100.0
# gridlines 0..100
for t in range(0,101,25):
    gx=xof(t); d.line([(gx,ytop-10),(gx,ytop+2*rowh+30)],fill=GRID,width=1)
    d.text((gx-8 if t<100 else gx-16,ytop+2*rowh+38),f"{t}%",font=tick,fill=MUT)
d.text((x0+ (x1-x0)/2 -120, ytop+2*rowh+64),"document-classification instability rate",font=small,fill=MUT)

rows=[
 ("Strategic prompt", '“would a trial lawyer flag this?”', 60,95, RED, "60–95%"),
 ("Textual-criteria prompt", '“contains an admission, a decision, …”', 3,7, GRN, "3–7%"),
]
for i,(name,desc,lo,hi,col,rng) in enumerate(rows):
    cy=ytop+i*rowh+30
    d.text((50,cy-2),name,font=lab,fill=INK)
    d.text((50,cy+26),desc,font=small,fill=MUT)
    # full track
    d.rounded_rectangle([x0,cy+2,x1,cy+40],radius=8,fill=(245,247,250))
    # band
    bl,br=xof(lo),xof(hi)
    if br-bl<10: br=bl+10
    d.rounded_rectangle([bl,cy+2,br,cy+40],radius=8,fill=col)
    lw=d.textlength(rng,font=lab)
    if br+14+lw > x1+4:
        d.text((bl-14-lw,cy+8),rng,font=lab,fill=col)   # label to the left when near the right edge
    else:
        d.text((br+14,cy+8),rng,font=lab,fill=col)

# footnote
fy=ytop+2*rowh+98
d.line([(50,fy),(W-50,fy)],fill=GRID,width=1)
d.text((50,fy+14),"Binary relevance (RELEVANT vs NOT) stayed ~98% stable — the instability is specific to",font=small,fill=INK)
d.text((50,fy+36),"unconstrained salience judgments, not to LLM classification in general.",font=small,fill=INK)

out=os.path.join(os.path.dirname(__file__),"instability.png")
img.save(out)
print("wrote",out,img.size)
