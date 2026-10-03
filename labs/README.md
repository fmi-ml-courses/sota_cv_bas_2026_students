# Практическая часть курса

Каждая лабораторная содержит `student.ipynb` с теоретическими сведениями, постановками задач и ячейками для самостоятельной реализации. Вычислительный профиль указан в README соответствующей лабораторной.

## Порядок работы

1. Откройте ноутбук в Google Colab и сохраните собственную копию.
2. Ознакомьтесь с постановками, зависимостями и требованиями к результатам. При необходимости подключите GPU.
3. Выполните подготовительные ячейки в режиме `FAST_DEV_RUN=True` и реализуйте ячейки заданий.
4. Выполните полный эксперимент, сохраните конфигурацию, контрольные точки и метрики в `artifacts/`. Перед завершением сессии скопируйте результаты в постоянное хранилище.
5. Подготовьте анализ результатов и проверьте критерии принятия каждой задачи.

## Навигация

| № | Лабораторная работа | Ноутбук | Google Colab |
|---:|---|---|---|
| 1 | Transfer learning и fine-tuning визуального backbone | [Задания](lab_01_transfer_learning/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_01_transfer_learning/student.ipynb) |
| 2 | Детекция и сегментация объектов | [Задания](lab_02_detection_segmentation/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_02_detection_segmentation/student.ipynb) |
| 3 | Foundation models для сегментации: SAM/SAM 2 | [Задания](lab_03_sam/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_03_sam/student.ipynb) |
| 4 | Self-supervised learning визуальных представлений | [Задания](lab_04_self_supervised/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_04_self_supervised/student.ipynb) |
| 5 | CLIP: zero-shot-классификация и мультимодальный поиск | [Задания](lab_05_clip_retrieval/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_05_clip_retrieval/student.ipynb) |
| 6 | Построение и оценка VLM-пайплайна | [Задания](lab_06_vlm_evaluation/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_06_vlm_evaluation/student.ipynb) |
| 7 | Автоматическая разметка изображений с помощью VLM | [Задания](lab_07_vlm_annotation/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_07_vlm_annotation/student.ipynb) |
| 8 | Active learning и human-in-the-loop | [Задания](lab_08_active_learning/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_08_active_learning/student.ipynb) |
| 9 | Fine-tuning VLM с помощью LLaMA-Factory | [Задания](lab_09_llamafactory_vlm_sft/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_09_llamafactory_vlm_sft/student.ipynb) |
| 10 | Эффективный fine-tuning VLM с помощью Unsloth | [Задания](lab_10_unsloth_efficiency/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_10_unsloth_efficiency/student.ipynb) |
| 11 | Alignment VLM по предпочтениям | [Задания](lab_11_vlm_alignment/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_11_vlm_alignment/student.ipynb) |
| 12 | Генерация синтетического датасета для VLM | [Задания](lab_12_synthetic_vlm_data/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_12_synthetic_vlm_data/student.ipynb) |
| 13 | Fine-tuning diffusion model | [Задания](lab_13_diffusion_lora/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_13_diffusion_lora/student.ipynb) |
| 14 | Управляемая генерация изображений | [Задания](lab_14_controlled_generation/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_14_controlled_generation/student.ipynb) |
| 15 | Flow matching и rectified flow | [Задания](lab_15_flow_matching/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_15_flow_matching/student.ipynb) |
| 16 | Визуальные токенизаторы и autoregressive image generation | [Задания](lab_16_visual_tokenization/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_16_visual_tokenization/student.ipynb) |
| 17 | Diffusion model с нуля и classifier-free guidance | [Задания](lab_17_ddpm_cfg/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_17_ddpm_cfg/student.ipynb) |
| 18 | Быстрые samplers и distillation компактной diffusion model | [Задания](lab_18_fast_samplers_distillation/student.ipynb) | [Открыть в Colab](https://colab.research.google.com/github/fmi-ml-courses/computer_vision_2026_students/blob/main/labs/lab_18_fast_samplers_distillation/student.ipynb) |

## Проверка перед сдачей

Перезапустите ядро и выполните ячейки по порядку. Все ячейки заданий должны содержать собственную реализацию вместо `NotImplementedError`. Проверьте наличие требуемых результатов, отсутствие ошибок исполнения и воспроизводимость эксперимента при зафиксированной конфигурации.

Доступность GPU и продолжительность бесплатной сессии Colab не гарантируются. Сокращённый запуск проверяет работоспособность частей программы, но не заменяет полный эксперимент и оценку его результатов.
