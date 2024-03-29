# SlicerDocker platform image
# (based on https://github.com/dit4c/dockerfile-dit4c-container-slicer)

ARG VERSION="latest"

FROM stevepieper/x11:${VERSION}
MAINTAINER pieper@isomics.com

RUN apt-get install -y libglu1-mesa
RUN apt-get install -y libpulse-dev libnss3 libglu1-mesa
RUN apt-get install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
RUN apt-get install -y libxcb-shape0 libxcb-xinerama0 libxcb-xinerama0-dev
RUN apt-get install -y libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0 libxcb-xkb-dev libxkbcommon-x11-dev
RUN apt-get install -y libgomp1
RUN apt-get install -y xvfb

# Slicer 4.5: http://download.slicer.org/bitstream/461634
# Slicer 4.8.1 http://download.slicer.org/bitstream/738960
# Slicer 4.10.1 http://download.slicer.org/bitstream/944596
# Slicer 4.10.2 http://download.slicer.org/bitstream/1023242
# Slicer 4.11.20200930 https://download.slicer.org/bitstream/60add70fae4540bf6a89bfb4

# Slicer-4.11.20210226
RUN alias python=python3
RUN SLICER_URL="https://download.slicer.org/bitstream/60add706ae4540bf6a89bf98" && \
  curl -k -v -s -L $SLICER_URL | tar xz -C /tmp && \
  mv /tmp/Slicer* /opt/slicer

# Download slicer extension 29738-linux-amd64-SlicerRT-gitce6eefe-2021-08-27.tar.gz
ARG SLICER_RT_FILE="/tmp/29738-linux-amd64-SlicerRT-gitce6eefe-2021-08-27.tar.gz"
RUN SLICER_RT_URL="https://slicer-packages.kitware.com/api/v1/item/60ae37baae4540bf6a89e617/download" && \
  curl -k -v -s -L $SLICER_RT_URL -o $SLICER_RT_FILE

ENV SLICER_RT_FILE=$SLICER_RT_FILE

# ARG SLICER_RT_INSTALL_SCIRPT=/tmp/install_slicer_rt.py
# RUN echo "import slicer; slicer.app.extensionsManagerModel().installExtension(sys.argv[1][0])" > $SLICER_RT_INSTALL_SCIRPT

# copy /opt after extraction slicer (/opt/slicer is extracted tar)
COPY /etc /etc
COPY /opt /opt
COPY /usr /usr

RUN chmod +x /usr/local/bin/start-xdummy

RUN chmod -R a+w /opt/slicer
# See http://na-mic.org/Mantis/view.php?id=4015
RUN su researcher -c "mkdir /home/researcher/Documents"
COPY /home/researcher/Documents /home/researcher/Documents
RUN chown -R researcher:researcher /home/researcher/Documents

COPY /home/researcher/.config /home/researcher/.config
RUN chown -R researcher:researcher /home/researcher/.config

RUN LNUM=$(sed -n '/launcher_item_app/=' /etc/tint2/panel.tint2rc | head -1) && \
  sed -i "${LNUM}ilauncher_item_app = /opt/slicer/slicer.desktop" /etc/tint2/panel.tint2rc

RUN echo "bash /tmp/shared/startup.sh" > /home/researcher/.xinitrc
RUN startx &
# Note that slicer is started automatically at login from /etc/skel/.xinitrc
# it will run whatever is in the user's home directory .slicerrc.py, so any
# users of this docker image have the option of putting arbitrary startup commands in that file.
#
# Note also you can set the SLICER_ARGUMENTS environment variable using the -e "SLICER_ARGUMENTS=<value>"
# on the docker run command line.
