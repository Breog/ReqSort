# 🧩 ReqSort: Requirement Classifier

[🇪🇸 Español](#español) · [🇬🇧 English](#english)

---

<a id="español"></a>

## 🇪🇸 Español

### 📘 Descripción

Aplicación web que **clasifica requisitos de software con IA** directamente en el navegador y **revisa su redacción** para proponer mejoras.

Escribes un requisito en inglés y el modelo lo clasifica en **funcional**, **no funcional**, una **regla de negocio** o una **restricción**, con la probabilidad de cada categoría. Además, la app señala los problemas de redacción más habituales (términos vagos, cláusulas de escape como *"if possible"*, enumeraciones abiertas como *"etc."*, comparaciones sin referencia, requisitos no funcionales sin métrica…) y sugiere una estructura recomendada.

El modelo es **DistilBERT** ajustado (*fine-tuning*) con un dataset propio de unos 1.400 requisitos y alcanza un **96 % de acierto** en un conjunto de test independiente de 400 frases, también propio. Se exporta a **ONNX cuantizado a 8 bits** (~64 MB) y se ejecuta en el dispositivo del visitante con **Transformers.js**: no hay backend y el texto nunca sale del navegador.

### 🌐 Aplicación

**Demo:** 

*La primera visita descarga el modelo (~64 MB); después queda en la caché del navegador.*

### ✨ Funcionalidades

* Clasificación en 4 categorías con la probabilidad de cada una.
* Varios requisitos a la vez (uno por línea).
* Revisión de la redacción basada en la norma **ISO/IEC/IEEE 29148**: resalta los términos problemáticos, explica cómo corregirlos y propone una plantilla según la categoría.
* Interfaz en español e inglés, modo oscuro automático y diseño adaptado a móvil.
* 100 % en el navegador: sin servidor, sin cuentas y sin enviar datos.

### 🧠 Cómo funciona

1. **Entrenamiento** (`training/train.py`): *fine-tuning* de `distilbert-base-uncased` con Hugging Face `Trainer` (división estratificada 80/20 para validación y selección de la mejor época por F1 macro).
2. **Evaluación** (`training/evaluate.py`): métricas sobre un test de 400 requisitos que no se solapa con el de entrenamiento.
3. **Exportación** (`training/export_onnx.py`): conversión a ONNX y cuantización int8, comprobando que la precisión se mantiene.
4. **Web** (`web/`): Transformers.js carga el modelo y clasifica en el navegador; `quality.js` revisa la redacción con reglas.

### 🛠️ Tecnologías

#### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers + Datasets
* scikit-learn, pandas, NumPy
* ONNX + ONNX Runtime (exportación y cuantización int8)

#### Web

* HTML, CSS y JavaScript (sin frameworks ni paso de compilación)
* Transformers.js (inferencia en el navegador con WebAssembly)

#### Despliegue

* Vercel (sitio estático)

### 📄 Licencia

**Partial Use License 1.0**

Entra en [LICENSE](LICENSE.txt) para ver la licencia completa.

---

<a id="english"></a>

## 🇬🇧 English

### 📘 Description

A web application that **classifies software requirements with AI** right in the browser and **reviews their wording** to suggest improvements.

You write a requirement in English and the model tells you whether it is **functional**, **non-functional**, a **business rule** or a **constraint**, with the probability of each category. The app also flags the most common wording problems (vague terms, escape clauses such as *"if possible"*, open-ended lists such as *"etc."*, comparisons without a baseline, non-functional requirements without a metric…) and suggests a recommended structure.

The model is **DistilBERT** fine-tuned on a custom dataset of about 1,400 requirements and reaches **96% accuracy** on a separate 400-sentence test set, also custom-made. It is exported to **8-bit quantized ONNX** (~64 MB) and runs on the visitor's device with **Transformers.js**: there is no backend and the text never leaves the browser.

### 🌐 Application

**Live Demo:** 

*The first visit downloads the model (~64 MB); after that it stays in the browser cache.*

### ✨ Features

* Classification into 4 categories with the probability of each one.
* Several requirements at once (one per line).
* Wording review based on **ISO/IEC/IEEE 29148**: highlights problematic terms, explains how to fix them and suggests a template for the category.
* Spanish and English interface, automatic dark mode and mobile-friendly layout.
* 100% in the browser: no server, no accounts and no data sent anywhere.

### 🧠 How it works

1. **Training** (`training/train.py`): fine-tuning of `distilbert-base-uncased` with the Hugging Face `Trainer` (stratified 80/20 validation split, best epoch selected by macro-F1).
2. **Evaluation** (`training/evaluate.py`): metrics on a 400-requirement test set that does not overlap with the training data.
3. **Export** (`training/export_onnx.py`): conversion to ONNX and int8 quantization, checking that accuracy is preserved.
4. **Web** (`web/`): Transformers.js loads the model and classifies in the browser; `quality.js` reviews the wording with rules.

### 🛠️ Technologies

#### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers + Datasets
* scikit-learn, pandas, NumPy
* ONNX + ONNX Runtime (export and int8 quantization)

#### Web

* HTML, CSS and JavaScript (no frameworks, no build step)
* Transformers.js (in-browser inference with WebAssembly)

#### Deployment

* Vercel (static site)

### 📄 License

**Partial Use License 1.0**

See the [LICENSE](LICENSE.txt) file for the complete license.
