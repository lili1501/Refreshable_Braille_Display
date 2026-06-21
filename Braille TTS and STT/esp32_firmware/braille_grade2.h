#pragma once
#include <Arduino.h>

// ============================================================
// Grade 2 (contracted) Braille translation
// ------------------------------------------------------------
// Converts a sequence of raw 6-bit braille cells (one per chord,
// bit 0 = dot 1 ... bit 5 = dot 6) into expanded English text.
//
// Handles:
//   - 26 letters (Grade 1 fallback)
//   - Strong groupsigns:  ch sh th wh gh ed er ou ow st ing ar
//   - Strong wordsigns:    and for of the with  + child shall this which out still
//   - Lower groupsigns (in-word): ea bb cc ff gg en in
//   - Alphabetic wordsigns (solo cell): but can do every from go have
//                                        just knowledge like more not people
//                                        quite rather so that us very will it you as
//   - Capital sign (dot 6) and Number sign (dots 3-4-5-6)
//   - Basic punctuation when a lower cell stands alone
//
// Pass the full set of cells that make up one "word" (the run of
// chords typed between spaces). The returned String is the text
// that word expands to.
// ============================================================
String grade2Word(const uint8_t* cells, int count);

// Translate a single cell ignoring word context (used for live echo
// of the most recent dot pattern). Returns "" if the cell is a
// prefix (capital/number) or unknown.
String grade2Cell(uint8_t pattern);
