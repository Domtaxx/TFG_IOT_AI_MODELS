#!/bin/bash

# 🔧 Output file
OUTFILE="slow_tshark_log.csv"

# ✅ Interface (change to wlan0, eth0, lo, etc. if needed)
INTERFACE="any"

# ✅ Capture MQTT traffic on port 1883
echo "[*] Starting TShark capture on interface: $INTERFACE"
echo "[*] Output file: $OUTFILE"

# 🔎 Capture command
tshark -i "$INTERFACE" -f "tcp port 1883" -T fields \
-e tcp.flags -e tcp.time_delta -e tcp.len \
-e mqtt.conack.flags -e mqtt.conack.flags.reserved -e mqtt.conack.flags.sp -e mqtt.conack.val \
-e mqtt.conflag.cleansess -e mqtt.conflag.passwd -e mqtt.conflag.qos -e mqtt.conflag.reserved \
-e mqtt.conflag.retain -e mqtt.conflag.uname -e mqtt.conflag.willflag -e mqtt.conflags \
-e mqtt.dupflag -e mqtt.hdrflags -e mqtt.kalive -e mqtt.len \
-e mqtt.topic -e mqtt.msg -e mqtt.msgid -e mqtt.msgtype -e mqtt.proto_len -e mqtt.protoname \
-e mqtt.qos -e mqtt.retain -e mqtt.sub.qos -e mqtt.suback.qos -e mqtt.ver \
-e mqtt.willmsg -e mqtt.willmsg_len -e mqtt.willtopic -e mqtt.willtopic_len \
-E header=y -E separator=, -E quote=d -E occurrence=f \
> "$OUTFILE"

echo "[+] TShark capture stopped. Output saved to $OUTFILE"
