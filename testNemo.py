import nemo
print(nemo.__version__)

import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tts

nlp_models = [model for model in dir(nemo_nlp.models) if model.endswith("Model")]
print(nlp_models)

asr_models = [model for model in dir(nemo_asr.models) if model.endswith("Model")]
print(asr_models)

citrinet = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_en_citrinet_512')

citrinet.summarize()