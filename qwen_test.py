PROMPT = ("This photo shows a utility pole. Find the pole tag or birthmark. "
          "Return only JSON with: tag_visible (true/false), all_text (exact text you can read), "
          "year (4 digits or null), confidence (0-1).")

def read_tag(path, max_side=1280):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))   # big photos eat VRAM; the crop handles fine detail
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", text, re.S)
    result = json.loads(m.group()) if m else {"raw": text}
    result["seconds"] = round(time.time() - t0, 1)
    return result

print("Full:   ", read_tag(IMG))
print("Cropped:", read_tag("tag_crop.jpg"))
