# RTNiFiOpenFace - demonstrating the use of OpenFace as an Apache NiFi WebSocket service

### Introduction

RTNiFiOpenFace demonstrates how to use OpenFace to recognize faces in an RTMQTT NiFi video flow. It is designed to work with the prebuilt Docker OpenFace image to simplify installation.

### Install and Run

Obtain the Docker image and clone RTNiFiOpenFace:

    cd ~/
    docker pull bamos/openface
    git clone git://github.com/richards-tech/RTNiFiOpenFace
    
Then, to run:

    sudo docker run -p 9000:9000 -p 8000:8000 -t -v /<absolute_path>/RTNiFiOpenFace:/root/src/RTNiFiOpenFace -i bamos/openface /bin/bash
    
Within the Docker execution environment, enter:

    cd /root/src/openface
    ../RTNiFiOpenFace/start-servers.sh
    
The RTNiFiOpenFace version of start-servers.sh works just as the standard version included with OpenFace except for two things. The first is that the WebSocket code detects if the message is from NiFi and processes it differently to those from the browser application. The second is that trained images are persisted and will be found in the RTNiFiOpenFace directory as two files called ofpeople.ini and ofimages.ini. When start-servers.sh starts the WebSocket server, it tries to load these files and, if it succeeds, will start using the data to recognize faces.

### Training

Training is via the browser application as described here - https://cmusatyalab.github.io/openface/demo-1-web/. 

### Using with NiFi

RTNiFiStreamProcessors (https://github.com/richards-tech/RTNiFiStreamProcessors) includes a custom processor, WebSocketProcessor, designed to interface with external servers (such as OpenFace) using a WebSocket client. So RTNiFiStreamProcessors should be installed for NiFi so that the flow can be built.

The basic pipeline starts with RTUVCCamMQTT (part of RTMQTT - https://github.com/richards-tech/RTMQTT) to capture video from a UVC webcam and publish the data as an MQTT topic. By default, this will be rtuvccam/video. An MQTT broker must be present - by default RTUVCCamMQTT assumes localhost but this can be changed.

NiFi can be configured to capture this flow by starting a flow with the GetMQTTProcessor from RTNiFiStreamProcessors. The broker address and topic need to be configured as does some sort of random client ID (it has to be unique is the only requirement). The MQTTMessage output of the GetMQTTProcessor should be connected to a WebSocketProcessor instance. This needs to be configured with the address of the WebSocker server. Assuming this is running on the same machine as RTNiFiOpenFace this would be "ws://localhost:9000".

The video returns from RTNiFiOpenFace still in RTMQTT video format so the output of WebSocketProcessor should be connected to an instance of RTMQTTVideoProcessor. This can save the video itself to a local file and output the metadata on another connection. In this case auto-terminate the Metadata relationship and connect the Video relationship to an instance of Putfile. PutFile needs to be configured with a directory to use for the video and also a conflict resolution strategy. If "append" is available, that can be used to build up files containing multiple jpeg images. Replace will just save the last image.

The easiest way to look at the images is to use NiFi's data provenance feature. There will be PutFile DROP entries and these contain the raw jpeg image. Viewing the content will display the image. If all is working, any recognized faces should have been annotated.

