import json, os
_here = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(_here, 'tokenizer_data.json'), encoding='utf-8'))
for mid, v in d.items():
    name = mid.split('/')[-1]
    print(f"{name}: class={v['tokenizer_class']}, vocab={v['vocab_size_full']}, "
          f"boundary={v['boundary_strategy']}, byte={v['has_byte_tokens']}, "
          f"special={v['special_token_count']}, examples={v['special_token_examples']}, "
          f"avgEN={v['avg_tokens_per_english_word']}, avgHE={v['avg_tokens_per_hebrew_word']}, "
          f"heRatio={v['he_en_ratio']}, chess={v['total_chess_tokens']}")
print()
for mid, v in d.items():
    name = mid.split('/')[-1]
    for span, toks in v['span_tokenizations'].items():
        print(f"{name}  [{span}]  -> {toks}")
