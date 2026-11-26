# Data Dictionary: SMS Spam Collection Dataset

**File:** `data/sms_spam_cleaned.csv`  
**Original Source:** UCI Machine Learning Repository (SMS Spam Collection Dataset, ID: 228)  
**Total Records:** 5,574 messages  
**Encoding:** UTF-8  

---

## 1. Column Specifications

| Column Name | Data Type | Permitted Values | Description |
|:---|:---|:---|:---|
| `label` | String (Categorical) | `'ham'`, `'spam'` | The classification label of the text message. `'ham'` indicates legitimate messages; `'spam'` indicates unsolicited spam/promotional messages. |
| `message` | String (Text) | Free-form string | The raw textual body of the short message service (SMS). Length ranges from 2 to 910 characters. |

---

## 2. Class Distribution Summary

| Class Label | Message Count | Percentage Share | Description |
|:---|:---:|:---:|:---|
| **ham** | 4,827 | 86.60% | Legitimate personal and transactional SMS communication |
| **spam** | 747 | 13.40% | Commercial advertisements, lottery alerts, phishing attempts |
| **Total** | **5,574** | **100.00%** | Cleaned dataset ready for Naive Bayes training |

---

## 3. Preprocessing Transformations Applied

1. **Lowercasing:** Converts all alphabetic characters to lowercase to eliminate case sensitivity (`FREE` $\rightarrow$ `free`).
2. **URL & Hyperlink Removal:** Regex patterns matching `http://`, `https://`, and `www.` prefixes are removed.
3. **Punctuation & Symbol Removal:** Standard ASCII punctuation marks (`!`, `?`, `$`, `%`, `,`, `.`) stripped.
4. **Digit Removal:** Standalone digits and phone numbers are normalized to avoid overfitting to arbitrary numbers.
5. **Whitespace Normalization:** Multiple consecutive spaces and newline characters are collapsed into single spaces.
6. **Tokenization Filter:** Words with a character length of 1 are discarded to retain meaningful semantic terms.

