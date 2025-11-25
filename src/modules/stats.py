"""Statistics tracking for Live Translator"""
import time

class TranslatorStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.chunks_processed = 0
        self.translations_made = 0
        self.hallucinations_filtered = 0
        self.total_processing_time = 0

    def add_chunk(self, processing_time, had_translation, was_hallucination):
        self.chunks_processed += 1
        self.total_processing_time += processing_time
        if had_translation:
            if was_hallucination:
                self.hallucinations_filtered += 1
            else:
                self.translations_made += 1