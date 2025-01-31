flags = {
    'a': '🇺🇸',
    'b': '🇬🇧',
    'e': '🇪🇸',
    'f': '🇫🇷',
    'h': '🇮🇳',
    'i': '🇮🇹',
    'j': '🇯🇵',
    'p': '🇧🇷',
    'z': '🇨🇳'
}

voices = {
    'a': [
        'af_alloy',
        'af_aoede',
        'af_bella',
        'af_heart',
        'af_jessica',
        'af_kore',
        'af_nicole',
        'af_nova',
        'af_river',
        'af_sarah',
        'af_sky',
        'am_adam',
        'am_echo',
        'am_eric',
        'am_fenrir',
        'am_liam',
        'am_michael',
        'am_onyx',
        'am_puck',
        'am_santa'],
    'b': [
        'bf_alice',
        'bf_emma',
        'bf_isabella',
        'bf_lily',
        'bm_daniel',
        'bm_fable',
        'bm_george',
        'bm_lewis'],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'],
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
    'z': [
        'zf_xiaobei',
        'zf_xiaoni',
        'zf_xiaoxiao',
        'zf_xiaoyi',
        'zm_yunjian',
        'zm_yunxi',
        'zm_yunxia',
        'zm_yunyang'
    ]
}

available_voices_str = ('\n'.join([f'  {flags[lang]} {", ".join(voices[lang])}' for lang in voices])
                        .replace(' af_sky,', '\n       af_sky,'))