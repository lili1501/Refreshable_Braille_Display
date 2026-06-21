#include "braille_grade2.h"

// ---- Cell pattern constants (bit 0 = dot 1 ... bit 5 = dot 6) ----
#define CELL_CAP   0b100000   // dot 6        -> capital sign
#define CELL_NUM   0b111100   // dots 3,4,5,6 -> number sign

// ---- Letter table: index a..z -> 6-bit pattern ----
static const uint8_t letterPat[26] = {
    0b000001, 0b000011, 0b001001, 0b011001, 0b010001, 0b001011,
    0b011011, 0b010011, 0b001010, 0b011010, 0b000101, 0b000111,
    0b001101, 0b011101, 0b010101, 0b001111, 0b011111, 0b010111,
    0b001110, 0b011110, 0b100101, 0b100111, 0b111010, 0b101101,
    0b111101, 0b110101
};

static char letterOf(uint8_t p) {
    for (int i = 0; i < 26; i++) if (letterPat[i] == p) return 'a' + i;
    return 0;
}

// ---- Number sign mode: letters a..j map to digits ----
static char numberOf(uint8_t p) {
    // a=1 b=2 c=3 d=4 e=5 f=6 g=7 h=8 i=9 j=0
    char l = letterOf(p);
    if (l == 'j') return '0';
    if (l >= 'a' && l <= 'i') return '1' + (l - 'a');
    return 0;
}

// ---- Pattern -> expansion lookup ----
struct G2Entry { uint8_t pat; const char* text; };

// Strong groupsigns / wordsigns: valid anywhere in a word.
static const G2Entry strongCells[] = {
    { 0b100001, "ch"  },   // dots 1,6
    { 0b100011, "gh"  },   // dots 1,2,6
    { 0b101001, "sh"  },   // dots 1,4,6
    { 0b111001, "th"  },   // dots 1,4,5,6
    { 0b110001, "wh"  },   // dots 1,5,6
    { 0b101011, "ed"  },   // dots 1,2,4,6
    { 0b111011, "er"  },   // dots 1,2,4,5,6
    { 0b110011, "ou"  },   // dots 1,2,5,6
    { 0b101010, "ow"  },   // dots 2,4,6
    { 0b001100, "st"  },   // dots 3,4
    { 0b101100, "ing" },   // dots 3,4,6
    { 0b011100, "ar"  },   // dots 3,4,5
    { 0b101110, "the" },   // dots 2,3,4,6
    { 0b101111, "and" },   // dots 1,2,3,4,6
    { 0b111111, "for" },   // dots 1,2,3,4,5,6
    { 0b110111, "of"  },   // dots 1,2,3,5,6
    { 0b111110, "with"},   // dots 2,3,4,5,6
};
static const int STRONG_N = sizeof(strongCells) / sizeof(strongCells[0]);

// Lower groupsigns: only valid between letters (not first cell of word).
static const G2Entry lowerCells[] = {
    { 0b000110, "bb" },    // dots 2,3
    { 0b010010, "cc" },    // dots 2,5
    { 0b110110, "gg" },    // dots 2,3,5,6
    { 0b100010, "en" },    // dots 2,6
    { 0b010100, "in" },    // dots 3,5
    { 0b000010, "ea" },    // dot 2
    { 0b010110, "ff" },    // dots 2,3,5
};
static const int LOWER_N = sizeof(lowerCells) / sizeof(lowerCells[0]);

// Punctuation when a lower cell stands alone / starts a word.
static const G2Entry punctCells[] = {
    { 0b000010, ","  },    // dot 2
    { 0b000110, ";"  },    // dots 2,3
    { 0b010010, ":"  },    // dots 2,5
    { 0b010110, "!"  },    // dots 2,3,5
    { 0b110100, "."  },    // dots 3,5,6 (legacy mapping)
    { 0b100110, "?"  },    // dots 2,3,6
    { 0b001100, "/"  },    // (st cell) rare standalone — keep groupsign instead
};
static const int PUNCT_N = sizeof(punctCells) / sizeof(punctCells[0]);

// Solo-cell whole-word signs (cell standing alone between spaces).
static const G2Entry soloWords[] = {
    // alphabetic wordsigns
    { 0b000011, "but" },       // b
    { 0b001001, "can" },       // c
    { 0b011001, "do" },        // d
    { 0b010001, "every" },     // e
    { 0b001011, "from" },      // f
    { 0b011011, "go" },        // g
    { 0b010011, "have" },      // h
    { 0b011010, "just" },      // j
    { 0b000101, "knowledge" }, // k
    { 0b000111, "like" },      // l
    { 0b001101, "more" },      // m
    { 0b011101, "not" },       // n
    { 0b001111, "people" },    // p
    { 0b011111, "quite" },     // q
    { 0b010111, "rather" },    // r
    { 0b001110, "so" },        // s
    { 0b011110, "that" },      // t
    { 0b100101, "us" },        // u
    { 0b100111, "very" },      // v
    { 0b111010, "will" },      // w
    { 0b101101, "it" },        // x
    { 0b111101, "you" },       // y
    { 0b110101, "as" },        // z
    // strong wordsigns
    { 0b100001, "child" },     // ch
    { 0b101001, "shall" },     // sh
    { 0b111001, "this" },      // th
    { 0b110001, "which" },     // wh
    { 0b110011, "out" },       // ou
    { 0b001100, "still" },     // st
    { 0b101110, "the" },       // the
    { 0b101111, "and" },       // and
    { 0b111111, "for" },       // for
    { 0b110111, "of" },        // of
    { 0b111110, "with" },      // with
};
static const int SOLO_N = sizeof(soloWords) / sizeof(soloWords[0]);

static const char* lookup(const G2Entry* t, int n, uint8_t p) {
    for (int i = 0; i < n; i++) if (t[i].pat == p) return t[i].text;
    return nullptr;
}

static String capitalize(const char* s, bool cap) {
    String out(s);
    if (cap && out.length() > 0) out.setCharAt(0, toupper(out[0]));
    return out;
}

String grade2Cell(uint8_t p) {
    if (p == CELL_CAP || p == CELL_NUM) return "";
    const char* s = lookup(strongCells, STRONG_N, p);
    if (s) return String(s);
    char l = letterOf(p);
    if (l) return String(l);
    return "";
}

String grade2Word(const uint8_t* cells, int count) {
    if (count <= 0) return "";

    // Solo whole-word sign: single cell that is a recognised wordsign.
    if (count == 1) {
        const char* w = lookup(soloWords, SOLO_N, cells[0]);
        if (w) return String(w);
    }

    String out;
    bool cap = false;
    bool num = false;

    for (int i = 0; i < count; i++) {
        uint8_t p = cells[i];

        if (p == CELL_CAP) { cap = true; continue; }
        if (p == CELL_NUM) { num = true; continue; }

        if (num) {
            char d = numberOf(p);
            if (d) { out += d; continue; }
            num = false;   // non-digit ends number mode, fall through as text
        }

        // Strong groupsign/wordsign — valid anywhere.
        const char* s = lookup(strongCells, STRONG_N, p);
        if (s) { out += capitalize(s, cap); cap = false; continue; }

        // Lower groupsign — only between letters (not first cell).
        if (i > 0) {
            const char* lg = lookup(lowerCells, LOWER_N, p);
            if (lg) { out += capitalize(lg, cap); cap = false; continue; }
        }

        // Letter.
        char l = letterOf(p);
        if (l) {
            if (cap) { l = toupper(l); cap = false; }
            out += l;
            continue;
        }

        // Standalone punctuation (e.g. comma, period typed mid-stream).
        const char* pn = lookup(punctCells, PUNCT_N, p);
        if (pn) { out += pn; continue; }
    }

    return out;
}
