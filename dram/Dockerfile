FROM doduo1.umcn.nl/xie/nov:py19dgl08

ENV PATH "$PATH:$CUDA_ROOT"
RUN echo "PATH=$PATH" > /etc/environment
RUN echo "CUDA_ROOT=$CUDA_ROOT" >> /etc/environment
RUN echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/environment


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm


COPY ./ /opt/algorithm/


ENTRYPOINT ["python3", "/opt/algorithm/process_pipeline.py"]


# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=emphysema_subtyping
LABEL nl.diagnijmegen.rse.algorithm.author="Weiyi Xie"

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=24G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=11G
