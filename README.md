Физико-математический институт

2026, осень

Безопасность автоматизированных систем (ФМИ)

Лектор - Барулина М.А.
Практика Ощепкова Н.В.

Основа для курса - https://github.com/fmi-ml-courses/computer_vision_2026_students


## Модуль 1. Визуальные представления и классические задачи (недели 1–5)

### Блок 1

**Лекция 1. Современное CV: от CNN к foundation models.**
- Повторение: свёртки, ResNet/ConvNeXt, ViT, patch embedding, inductive bias.
- Transfer learning: linear probing, частичная разморозка, полный fine-tuning; learning rate для разных слоёв.
- Протокол эксперимента: разбиение без утечек (группы, дубликаты, связанные объекты), seed, конфиги, чекпоинты, журнал метрик.
- Обзор курса и сквозного проекта.

**Практика 1. Transfer learning и fine-tuning визуального backbone** (лаб. 1).
- Настройка Colab: Google Drive, `artifacts/`, продолжение после обрыва сессии, режим `FAST_DEV_RUN=True`.
- Обязательная часть: разбиение train/val/test; три режима (замороженный backbone, частичная разморозка, полный fine-tuning); сравнение CNN и ViT по качеству, времени, памяти и скорости; confusion matrix и разбор 20 ошибок; проверка на простом domain shift (освещение, шум).
- Результат: training script, таблица экспериментов, лучшая модель, error analysis.

### Блок 2

**Лекция 2. Детекция и сегментация объектов.**
- Постановки: детекция, semantic, instance и panoptic segmentation.
- Архитектуры: одностадийные (YOLO-семейство), двухстадийные (Faster/Mask R-CNN), трансформерные (DETR, RT-DETR), U-Net-подобные сегментаторы.
- Формат COCO, аудит разметки, малые объекты и редкие классы, аугментации.
- Метрики: mAP@[.5:.95], AP по размерам, IoU, Dice; типология FP/FN.

**Практика 2. Детекция и сегментация** (лаб. 2).
- Обязательная часть: перевод разметки в COCO-формат; визуальный аудит выборки, распределения классов и размеров; fine-tuning современной модели; эксперимент с аугментациями; метрики по классам и размерам; разбор FP/FN.
- Результат: модель, визуализации предсказаний, отчёт с метриками и типологией ошибок.

### Блок 3

**Лекция 3. Foundation models для сегментации и open-vocabulary.**
- SAM/SAM 2: image encoder, prompt encoder, mask decoder; point, box и mask prompts; работа с видео в SAM 2.
- Облегчённые версии (MobileSAM, EfficientSAM), адаптация через лёгкие головы.
- Open-vocabulary детекция (Grounding DINO, OWL-ViT) как генератор prompts для SAM.
- SAM как инструмент ускорения разметки: качество предразметки, стоимость ручной правки.

**Практика 3. SAM/SAM 2: promptable segmentation** (лаб. 3).
- Обязательная часть: zero-shot-маски по point/box prompts; автоматические prompts из разметки или детектора из П2; сравнение с supervised baseline; обучение mask refinement head на замороженном encoder MobileSAM; оценка доли масок, требующих значительной правки.
- Результат: segmentation pipeline и сравнение zero-shot, adapted и supervised подходов.

### Блок 4

**Лекция 4. Self-supervised learning визуальных представлений.**
- Contrastive learning (SimCLR, MoCo), самодистилляция без негативов (BYOL, DINO/DINOv2), masked image modeling (MAE, iBOT).
- Оценка представлений: k-NN, linear probing, few-shot; кривые по объёму разметки.
- Визуализация эмбеддингов (UMAP/t-SNE), паразитные признаки и shortcut learning.

**Практика 4. Self-supervised visual embeddings** (лаб. 4).
- Обязательная часть: признаки DINO-подобного энкодера; k-NN и linear probing; linear probing на 1%, 10%, 100% разметки; визуализация пространства признаков и ближайших соседей; поиск случаев группировки по нежелательному признаку.
- Дополнительная часть: собственный небольшой contrastive или MIM-baseline либо адаптация энкодера на неразмеченных данных.
- Результат: набор эмбеддингов и отчёт о качестве представлений при разном объёме разметки.

### Блок 5

**Лекция 5. CLIP и совместные пространства изображений и текстов.**
- Contrastive image-text pretraining: InfoNCE, температура, масштаб данных; CLIP, SigLIP, OpenCLIP.
- Zero-shot классификация, prompt engineering и ансамбли шаблонов.
- Мультимодальный поиск: индексы, Recall@K, mAP.
- Доменная адаптация: linear/projection head, prompt learning (CoOp), LoRA; риск забывания.

**Практика 5. CLIP: zero-shot и мультимодальный поиск** (лаб. 5).
- Обязательная часть: zero-shot по нескольким шаблонам prompts; индекс и text-to-image поиск; доменные пары «изображение — описание»; дообучение projection head или LoRA-адаптера с contrastive loss; сравнение по accuracy и Recall@K, удачные и неудачные примеры.
- Результат: text-to-image retrieval pipeline и оценка эффекта адаптации.

## Модуль 2. Vision-language модели и данные (недели 6–10)

### Блок 6

**Лекция 6. Архитектура и оценка VLM.**
- Схема «визуальный энкодер → проектор → LLM»: LLaVA, Qwen-VL, InternVL, SmolVLM; обработка высокого разрешения, число визуальных токенов.
- Квантизация 4-bit NF4, бюджет VRAM, memory probe и переход 4B → 2B.
- Оценка: captioning, VQA, structured extraction; бенчмарки и их ограничения; LLM-as-judge.
- Типичные ошибки: hallucinations, OCR, пространственные отношения, подсчёт объектов.

**Практика 6. Построение и оценка VLM-пайплайна** (лаб. 6).
- Обязательная часть: запуск VLM 2–4B в 4-bit; собственный evaluation set с категориями сложности; единый inference pipeline с логированием prompt, параметров генерации, времени и памяти; сравнение prompts, разрешений и способов декодирования; ручная и автоматическая оценка с разметкой типов ошибок.
- Результат: benchmark VLM с воспроизводимыми prompts, метриками и каталогом ошибок.
- Здесь студенты выбирают домен, который будет использоваться в П7–П10.

### Блок 7

**Лекция 7. Data-centric CV: автоматическая разметка и active learning.**
- Схемы разметки, JSON Schema, constrained/structured generation, повтор при невалидном ответе.
- Контроль качества: self-consistency, confidence proxy, проверка второй моделью, стратифицированная ручная проверка, agreement.
- Active learning: uncertainty, diversity (core-set), disagreement; annotation budget; cold start.
- Human-in-the-loop, provenance и версионирование датасета.

**Практика 7. Автоматическая разметка VLM и active learning** (лаб. 7 + сокращённая лаб. 8).
- Обязательная часть: схема разметки и JSON Schema; разметка выборки VLM с сохранением сырых ответов; фильтрация (self-consistency или вторая модель); ручная проверка стратифицированной подвыборки с precision/recall по полям; 3 итерации active learning с uncertainty sampling против случайного отбора при одинаковом бюджете; учёт источника каждой метки.
- Дополнительная часть: вторая стратегия (diversity), сравнение модели на автоматической разметке с моделью на малой ручной выборке.
- Результат: версионированный датасет, отчёт о качестве разметки, learning curves.

### Блок 8

**Лекция 8. Parameter-efficient fine-tuning VLM.**
- SFT мультимодальных моделей: instruction-формат, chat templates, специальные visual tokens, маскирование loss.
- LoRA и QLoRA: rank, alpha, целевые модули; NF4, double quantization, paged optimizers.
- Экономия памяти: batch size 1, gradient accumulation, gradient checkpointing, memory-efficient attention.
- Инструменты: LLaMA-Factory, Unsloth, Transformers/PEFT; воспроизводимость конфигов.
- Оценка после адаптации: OOD-примеры, катастрофическое забывание.

**Практика 8. QLoRA fine-tuning VLM** (лаб. 9 + элементы лаб. 10).
- Обязательная часть: перевод данных из П7 в instruction-формат LLaMA-Factory и проверка visual tokens; конфигурация 4-bit QLoRA; обучение с сохранением адаптера, логов и версии базовой модели; сравнение base и fine-tuned на отложенной и OOD-выборке; проверка сохранения общих способностей.
- Дополнительная часть: тот же запуск через Unsloth с замером VRAM, времени и throughput; два значения rank LoRA; график «качество — время — память».
- Результат: LoRA-адаптер, воспроизводимый конфиг, evaluation report.

### Блок 9

**Лекция 9. Alignment VLM по предпочтениям.**
- От SFT к preference optimization: RLHF (кратко), reward model, DPO, его варианты (IPO, ORPO, KTO).
- Preference данные для VLM: chosen/rejected, сбор «плохих» ответов, контроль длины и стиля.
- Борьба с hallucinations через предпочтения; метрики win rate, hallucination rate, соблюдение формата.
- Побочные эффекты: reward hacking, многословие, деградация исходной задачи.

**Практика 9. DPO для VLM** (лаб. 11).
- Обязательная часть: preference dataset (chosen и галлюцинирующие или плохо отформатированные rejected); проверка баланса типов предпочтений; DPO только для LoRA на 4-bit модели (по умолчанию 2B); слепое сравнение base, SFT и aligned; win rate, hallucination rate, формат, качество исходной задачи.
- Результат: preference dataset, aligned adapter, анализ пользы и побочных эффектов.

### Блок 10

**Лекция 10. Синтетические данные для CV и VLM.**
- Источники синтетики: программный рендеринг, compositing, генеративные модели; domain randomization.
- Конвейер: пространство сцен → спецификации (LLM) → изображения → проверка VLM → фильтрация → ручной аудит.
- Provenance (prompt, seed, параметры сцены), дедупликация, отбор сложных примеров.
- Domain gap, смеси real/synthetic, оценка только на реальном test set.

**Практика 10. Синтетический датасет для VLM** (лаб. 12).
- Обязательная часть: описание пространства сцен, включая редкие случаи; генерация спецификаций и вопросов; получение изображений рендерингом или compositing в пределах одной сессии Colab; проверка соответствия спецификации через VLM и формирование QA-пар; фильтрация и ручная проверка части данных; обучение на смесях 100/0, 50/50, 25/75 и оценка на реальном test set.
- Результат: синтетический датасет с provenance и оценка его пользы.

## Модуль 3. Генеративные модели изображений (недели 11–14)

### Блок 11

**Лекция 11. Визуальные токенизаторы и авторегрессионная генерация.**
- Обзор генеративных моделей: правдоподобие, латентные переменные, задача «генеративной трилеммы».
- VAE, VQ-VAE, VQGAN: codebook, commitment loss, straight-through estimator, коллапс кодов.
- Длина последовательности и spatial compression; токенизаторы в современных VLM и генераторах.
- Авторегрессионные приоры: ImageGPT, VQGAN-Transformer, VAR (next-scale prediction); temperature, top-k, top-p.

**Практика 11. Visual tokenizer и авторегрессионный prior** (лаб. 16).
- Обязательная часть: VQ-VAE на MNIST/CIFAR-10 (≤64×64); реконструкции; статистика codebook (доля используемых кодов, perplexity, «мёртвые» токены); один эксперимент с размером codebook или степенью сжатия; маленький decoder-only Transformer с условием по классу; сравнение стратегий sampling.
- Результат: tokenizer, AR prior, отчёт о влиянии дискретизации и sampling.

### Блок 12

**Лекция 12. Диффузионные модели.**
- Прямой и обратный процесс, DDPM, вариационная нижняя граница, предсказание noise, x0 и velocity.
- Score matching и связь со SDE (на уровне идей).
- Архитектуры: U-Net с timestep embedding, DiT.
- Условная генерация: classifier guidance и classifier-free guidance; компромисс fidelity/diversity; noise schedules.

**Практика 12. Diffusion model с нуля и classifier-free guidance** (лаб. 17).
- Обязательная часть: прямой процесс зашумления на MNIST/Fashion-MNIST/CIFAR-10; компактная U-Net; DDPM sampling с визуализацией промежуточных состояний; conditioning по классу и classifier-free dropout; генерации при нескольких guidance scale; влияние noise schedule или параметризации.
- Важно: сохранить чекпоинт для П13.
- Результат: компактная conditional diffusion model и анализ CFG.

### Блок 13

**Лекция 13. Быстрая генерация: samplers, flow matching, distillation.**
- DDIM как детерминированный sampler; ODE-формулировка; DPM-Solver, predictor-corrector.
- Flow matching и rectified flow: conditional probability paths, векторное поле, выпрямление траекторий.
- Distillation: progressive distillation, consistency models; сравнение при равном числе вызовов модели (NFE).
- Pareto frontier «качество — время».

**Практика 13. Ускорение генерации и flow matching** (лаб. 18 + лаб. 15).
- Обязательная часть: baseline DDPM для модели из П12 с фиксированными noise tensors; DDIM при нескольких бюджетах шагов; компактная flow matching модель на том же датасете при близком вычислительном бюджете; сравнение качества, diversity и скорости в зависимости от числа шагов; Pareto frontier.
- Дополнительная часть: готовый быстрый solver (DPM-Solver); consistency или progressive distillation на 2D-данных/MNIST; второй вариант probability path.
- Результат: benchmark DDPM, DDIM и flow matching с воспроизводимыми замерами.

### Блок 14

**Лекция 14. Latent diffusion, управляемая генерация и оценка генеративных моделей.**
- Latent diffusion и Stable Diffusion: VAE, text encoder, U-Net, scheduler.
- Адаптация: LoRA, DreamBooth, textual inversion; риски переобучения и memorization.
- Управление: ControlNet (edges, depth, pose, segmentation), IP-Adapter, inpainting.
- Оценка: FID/KID, CLIP score, diversity-метрики, слепая ручная оценка; проверка соблюдения условия.
- Этика и лицензии: авторские права, deepfakes, маркировка синтетики. Подведение итогов курса.

**Практика 14. LoRA для diffusion model и управляемая генерация** (лаб. 13 + лаб. 14).
- Обязательная часть: малый очищенный набор изображений с captions; baseline-генерации SD 1.5 с фиксированными seeds; LoRA-fine-tuning в FP16 с gradient checkpointing; сравнение checkpoints через CLIP similarity, diversity и слепую оценку; проверка memorization поиском ближайших обучающих изображений; генерация с готовым ControlNet при нескольких значениях силы контроля и guidance scale.
- Дополнительная часть: синтетический датасет, где control signal служит разметкой, и дообучение детектора или сегментатора из П2 на смеси данных.
- Результат: diffusion-адаптер, сравнительные генерации, оценка управляемости.

## Сводная таблица

| Неделя | Лекция | Практика | Исходные лаб. | Ключевой артефакт |
|---|---|---|---|---|
| 1 | Современное CV, transfer learning | Fine-tuning backbone | 1 | Training script, таблица экспериментов |
| 2 | Детекция и сегментация | Детектор/сегментатор | 2 | Модель, mAP/IoU, типология ошибок |
| 3 | SAM и open-vocabulary | Promptable segmentation | 3 | Pipeline SAM, сравнение режимов |
| 4 | Self-supervised learning | SSL embeddings | 4 | Эмбеддинги, кривые 1/10/100% |
| 5 | CLIP | Zero-shot и поиск | 5 | Retrieval pipeline, Recall@K |
| 6 | Архитектура и оценка VLM | VLM benchmark | 6 | Evaluation set, каталог ошибок |
| 7 | Авторазметка и active learning | Разметка VLM + AL | 7, 8 | Версионированный датасет, learning curves |
| 8 | PEFT для VLM | QLoRA fine-tuning | 9, 10 | LoRA-адаптер, конфиг |
| 9 | Alignment | DPO | 11 | Preference dataset, aligned adapter |
| 10 | Синтетические данные | Синтетика для VLM | 12 | Датасет с provenance, смеси real/synth |
| 11 | Токенизаторы и AR-генерация | VQ-VAE + AR prior | 16 | Tokenizer, prior, статистика codebook |
| 12 | Диффузия | DDPM + CFG с нуля | 17 | Conditional diffusion model |
| 13 | Samplers, flow matching | Ускорение генерации | 15, 18 | Pareto frontier |
| 14 | Latent diffusion, контроль, оценка | LoRA SD + ControlNet | 13, 14 | Diffusion-адаптер, оценка управляемости |

## Организация вычислений

- Все практики выполняются в бесплатном Google Colab; платные API и полный fine-tuning VLM не используются.
- Каждый ноутбук сначала запускается с `FAST_DEV_RUN=True`, затем в полном режиме с сохранением чекпоинтов и метрик в `artifacts/` на Google Drive.
- Для VLM перед запуском выполняется memory probe; при нехватке VRAM ноутбук переключается с 4B на 2B без изменения задания.
- Практики 7–10 рекомендуется выполнять в одном домене, выбранном на П6; П12 и П13 используют общий чекпоинт.
- Обязательная часть каждой практики рассчитана на аудиторное занятие плюс 4–6 часов самостоятельной работы; дополнительная часть даёт бонусные баллы.

## Требования к сдаче

Для каждой практики:

1. Код, конфигурации и зафиксированные зависимости.
2. Инструкция воспроизводимого запуска.
3. Параметры экспериментов, метрики и ключевые артефакты.
4. Краткий отчёт: baseline, результат, анализ ошибок, выводы.
5. Сведения о данных, лицензиях, базовых моделях и вычислительных ресурсах.

Сдача — защита у преподавателя в течение двух недель после занятия (короткая демонстрация и 2–3 вопроса по теории соответствующей лекции).
