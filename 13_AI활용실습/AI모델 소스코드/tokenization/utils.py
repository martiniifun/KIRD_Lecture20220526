import re
from konlpy.tag import Mecab

SPACE_PREFIX_4_MECAB = "[SP]"
JOIN_PREFIX_4_MECAB = "[J]"

def detokenize(token_list, mecab=False, spacing=False, joining=False, keep_space=True):
    text = " ".join(token_list)

    # wordpieced(##) -> basic tokens
    detokenized = text.replace(' ##', '')
    detokenized = detokenized.replace('##', '')

    if not mecab:
        detokenized = re.sub(r'\[unused[0-9]{1,3}\]', ' ', detokenized)

        # keep space with mecab
        if keep_space:
            mecab = Mecab()
            pos_list = mecab.pos(detokenized, flatten=True)
            morp_list = [morp for morp, tag in pos_list if morp != '']
            return morp_list

        return detokenized.split()

    # remove pos
    detokenized = re.sub(r'/[A-Z]{2,2}', '', detokenized).split()

    SPACE_ALT = " " if keep_space else ""

    if spacing:
        detokenized = [
            token.replace(SPACE_PREFIX_4_MECAB, SPACE_ALT, 1) + ' ' if token.startswith(SPACE_PREFIX_4_MECAB) else token
            for token in detokenized]
    elif joining:
        detokenized = [
            token.replace(JOIN_PREFIX_4_MECAB, SPACE_ALT, 1) if token.startswith(JOIN_PREFIX_4_MECAB) else ' ' + token
            for token in detokenized]

    # Update tokens with space
    detokenized = "".join(detokenized).split()

    return detokenized
