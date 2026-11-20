
import os
import io
import json
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, render_template_string
)
from openai import OpenAI

# =========================================
# 1. API 설정
# =========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_API_KEY = os.getenv("KAKAO_REST_API_KEY")

if not OPENAI_API_KEY or not KAKAO_API_KEY:
    raise RuntimeError(
        ".env 파일에 OpenAI와 Kakao API 키를 입력하세요."
    )

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================================
# 2. OpenAI 음성인식
# =========================================

def transcribe_audio(upload):

    audio_bytes = upload.read()

    if not audio_bytes:
        raise ValueError("음성 파일이 비어 있습니다.")

    mime = (upload.mimetype or "").split(";")[0]

    if mime in ("audio/webm", "video/webm"):
        filename = "voice.webm"
        mime = "audio/webm"
    elif mime in ("audio/mp4", "video/mp4"):
        filename = "voice.mp4"
        mime = "audio/mp4"
    elif mime == "audio/ogg":
        filename = "voice.ogg"
    else:
        raise ValueError(
            "지원하지 않는 음성 파일 형식입니다."
        )

    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=(filename, audio_bytes, mime),
        language="ko",
        prompt=(
            "한국어 차량 내비게이션 명령입니다. "
            "서울역, 강남역, 녹색병원, "
            "동양미래대학교 등의 장소명이 등장합니다."
        )
    )

    return result.text.strip()


# =========================================
# 3. GPT 목적지 추출
# =========================================

def extract_destination(text):

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""
        당신은 한국어 차량 내비게이션 명령 분석기입니다.

        사용자의 음성인식 문장에서 목적지를 추출하세요.

        규칙:
        1. 장소명을 임의로 축약하거나 변경하지 마세요.
        2. 이동 요청이 아니면 null을 반환하세요.
        3. 서울, 부산 같은 도시 이름만 있으면
           목적지가 불명확하므로 null을 반환하세요.
        4. 특정 역, 병원, 학교, 건물 등은 추출하세요.
        5. 음성인식 오류로 장소가 불명확하면
           추측하지 말고 null을 반환하세요.

        예시:
        "서울역으로 가자" -> "서울역"
        "강남역까지 가줘" -> "강남역"
        "녹색병원 안내해줘" -> "녹색병원"
        "동양미래대학교로 가자" -> "동양미래대학교"
        "서울 가자" -> null
        "오늘 날씨 어때?" -> null

        JSON 형식으로만 응답하세요.
        """,
        input=text,
        text={
            "format": {
                "type": "json_schema",
                "name": "navigation_command",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": ["string", "null"]
                        }
                    },
                    "required": ["destination"],
                    "additionalProperties": False
                }
            }
        }
    )

    data = json.loads(response.output_text)
    destination = data["destination"]

    if destination is None:
        return None

    if not isinstance(destination, str):
        return None

    destination = destination.strip()

    if not destination or len(destination) > 100:
        return None

    return destination


# =========================================
# 4. 카카오 장소 검색
# =========================================

def search_kakao(destination):

    url = (
        "https://dapi.kakao.com/"
        "v2/local/search/keyword.json"
    )

    response = requests.get(
        url,
        headers={
            "Authorization": f"KakaoAK {KAKAO_API_KEY}"
        },
        params={
            "query": destination,
            "size": 5
        },
        timeout=10
    )

    # 권한 오류 확인
    if response.status_code != 200:
        app.logger.error(
            "Kakao API error: HTTP %s / %s",
            response.status_code,
            response.text[:500]
        )
        response.raise_for_status()

    documents = response.json().get("documents", [])
    places = []

    for item in documents:
        places.append({
            "name": item["place_name"],
            "address": (
                item.get("road_address_name")
                or item.get("address_name")
                or ""
            ),
            "latitude": float(item["y"]),
            "longitude": float(item["x"])
        })

    return places


# =========================================
# 5. 티맵 링크 생성
# =========================================

def make_tmap_link(place):

    return "tmap://route?" + urlencode({
        "goalname": place["name"],
        "goalx": place["longitude"],
        "goaly": place["latitude"]
    })


# =========================================
# 6. HTML 웹페이지
# =========================================

HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>AI 음성 내비게이션</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    background: #f1f5f9;
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}

.container {
    background: white;
    width: 100%;
    max-width: 430px;
    border-radius: 22px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 8px 30px #00000012;
}

h2 {
    margin: 0 0 12px;
}

.desc {
    color: #64748b;
    font-size: 15px;
}

.mic {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: #eff6ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
    margin: 28px auto;
}

.mic.recording {
    background: #fee2e2;
    animation: pulse 1.2s infinite;
}

@keyframes pulse {
    50% { transform: scale(1.06); }
}

button, .tmap {
    width: 100%;
    display: block;
    padding: 16px;
    border: none;
    border-radius: 12px;
    font-size: 17px;
    font-weight: bold;
    cursor: pointer;
    text-decoration: none;
}

button {
    background: #2563eb;
    color: white;
}

button:disabled {
    opacity: 0.55;
    cursor: default;
}

.tmap {
    margin-top: 15px;
    background: #059669;
    color: white;
}

#status {
    margin-top: 20px;
    line-height: 1.7;
    white-space: pre-line;
}

#transcript {
    margin-top: 14px;
    padding: 14px;
    border-radius: 10px;
    background: #f1f5f9;
    white-space: pre-wrap;
    text-align: left;
    display: none;
}

.error {
    color: #dc2626;
}

.place {
    padding: 14px;
    margin-top: 12px;
    text-align: left;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    background: white;
    cursor: pointer;
}

.place small {
    display: block;
    margin-top: 5px;
    color: #64748b;
}

.place.selected {
    border: 2px solid #2563eb;
    background: #eff6ff;
}

footer {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 24px;
}
</style>
</head>

<body>
<div class="container">

    <h2>AI 음성 내비게이션</h2>
    <p class="desc">목적지를 말씀하세요.</p>

    <div id="micIcon" class="mic">🎤</div>

    <button id="mainButton" onclick="handleButton()">
        말하기
    </button>

    <div id="status" aria-live="polite">
        버튼을 누르면 녹음이 시작됩니다.
    </div>

    <div id="transcript"></div>

    <div id="places"></div>

    <a id="tmap" class="tmap" style="display:none">
        TMAP 가기
    </a>

    <footer>
        장소 검색: Kakao Local API
    </footer>

</div>

<script>

let recorder = null;
let stream = null;
let chunks = [];
let state = "idle";

const button = document.getElementById("mainButton");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const placesEl = document.getElementById("places");
const tmap = document.getElementById("tmap");
const micIcon = document.getElementById("micIcon");

function showStatus(message, error = false) {
    statusEl.textContent = message;
    statusEl.className = error ? "error" : "";
}

function setState(next) {
    state = next;

    if (next === "recording") {
        button.textContent = "입력 완료";
        button.disabled = false;
        micIcon.classList.add("recording");
    } else if (next === "processing") {
        button.textContent = "분석 중...";
        button.disabled = true;
        micIcon.classList.remove("recording");
    } else {
        button.textContent = "다시 말하기";
        button.disabled = false;
        micIcon.classList.remove("recording");
    }
}

function resetResult() {
    placesEl.innerHTML = "";
    tmap.style.display = "none";
    tmap.removeAttribute("href");
    transcriptEl.style.display = "none";
    transcriptEl.textContent = "";
}

function stopMicrophone() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

function getRecordingType() {
    const types = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4"
    ];

    return types.find(type =>
        MediaRecorder.isTypeSupported(type)
    ) || "";
}

async function handleButton() {

    if (state === "recording") {
        finishRecording();
        return;
    }

    if (state === "processing") return;

    await startRecording();
}

async function startRecording() {

    resetResult();

    if (!window.isSecureContext) {
        showStatus(
            "HTTPS 또는 localhost에서 접속해야 합니다.",
            true
        );
        return;
    }

    if (!navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia ||
        !window.MediaRecorder) {

        showStatus(
            "이 브라우저에서 녹음 기능을 사용할 수 없습니다.",
            true
        );
        return;
    }

    try {
        showStatus("마이크 권한을 확인하고 있습니다.");

        stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        const mimeType = getRecordingType();

        if (!mimeType) {
            throw new Error(
                "지원되는 음성 녹음 형식이 없습니다."
            );
        }

        recorder = new MediaRecorder(stream, {
            mimeType: mimeType
        });

        chunks = [];

        recorder.ondataavailable = event => {
            if (event.data && event.data.size > 0) {
                chunks.push(event.data);
            }
        };

        recorder.onerror = event => {
            console.error("Recorder error:", event);
            stopMicrophone();
            setState("idle");
            showStatus(
                "녹음 중 오류가 발생했습니다. 다시 말씀해 주세요.",
                true
            );
        };

        recorder.onstop = async () => {
            stopMicrophone();

            if (state !== "processing") return;

            if (chunks.length === 0) {
                setState("idle");
                showStatus(
                    "음성이 녹음되지 않았습니다. 다시 말씀해 주세요.",
                    true
                );
                return;
            }

            const blob = new Blob(chunks, {
                type: mimeType
            });

            await sendAudio(blob);
        };

        recorder.start(250);
        setState("recording");

        showStatus(
            "녹음 중입니다.\n" +
            "목적지를 말씀한 뒤 입력 완료를 누르세요."
        );

    } catch (error) {
        stopMicrophone();
        setState("idle");

        const message =
            error.name === "NotAllowedError"
                ? "마이크 권한이 차단되었습니다."
                : "마이크 오류: " + error.message;

        showStatus(message, true);
    }
}

function finishRecording() {

    if (!recorder ||
        recorder.state !== "recording") {
        return;
    }

    setState("processing");
    showStatus("음성을 분석하고 있습니다...");

    recorder.stop();
}

async function sendAudio(blob) {

    const mimeType = blob.type.split(";")[0];

    let extension;

    if (mimeType === "audio/mp4") {
        extension = "mp4";
    } else {
        extension = "webm";
    }

    const form = new FormData();

    form.append(
        "audio",
        blob,
        "voice." + extension
    );

    try {
        const response = await fetch("/navigate", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (data.transcript) {
            transcriptEl.textContent =
                "인식된 문장: " + data.transcript;

            transcriptEl.style.display = "block";
        }

        if (!response.ok || !data.success) {
            showStatus(
                data.message || "다시 말씀해 주세요.",
                true
            );
            return;
        }

        showPlaces(data.places);

    } catch (error) {

        showStatus(
            "서버 연결에 실패했습니다. 다시 시도하세요.",
            true
        );

    } finally {
        setState("idle");
    }
}

function showPlaces(places) {

    placesEl.innerHTML = "";
    tmap.style.display = "none";

    showStatus(
        "목적지를 찾았습니다.\n" +
        "정확한 장소를 선택하세요."
    );

    places.forEach(place => {

        const item = document.createElement("div");
        item.className = "place";

        const name = document.createElement("strong");
        name.textContent = place.name;

        const address = document.createElement("small");
        address.textContent = place.address;

        item.append(name, address);

        item.onclick = () => {

            document.querySelectorAll(".place")
                .forEach(el =>
                    el.classList.remove("selected")
                );

            item.classList.add("selected");

            tmap.href = place.url;
            tmap.style.display = "block";
        };

        placesEl.appendChild(item);
    });

    if (places.length === 1) {
        placesEl.firstElementChild.click();
    }
}

</script>
</body>
</html>
"""


# =========================================
# 7. Flask
# =========================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/navigate", methods=["POST"])
def navigate():

    upload = request.files.get("audio")

    if not upload:
        return jsonify(
            success=False,
            message="음성 파일이 없습니다."
        ), 400

    transcript = ""

    try:
        # 음성 -> 텍스트
        transcript = transcribe_audio(upload)

        print("[음성인식]", transcript)

        if not transcript:
            return jsonify(
                success=False,
                message="음성을 인식하지 못했습니다. "
                        "다시 말씀해 주세요."
            )

        # GPT 목적지 추출
        destination = extract_destination(transcript)

        print("[GPT 목적지]", destination)

        if not destination:
            return jsonify(
                success=False,
                transcript=transcript,
                message="목적지를 이해하지 못했습니다. "
                        "다시 말씀해 주세요."
            )

        # 카카오 장소 검색
        places = search_kakao(destination)

        print("[카카오 검색 결과 수]", len(places))

        if not places:
            return jsonify(
                success=False,
                transcript=transcript,
                message="해당 목적지를 찾지 못했습니다. "
                        "다시 말씀해 주세요."
            )

        for place in places:
            place["url"] = make_tmap_link(place)

        return jsonify(
            success=True,
            transcript=transcript,
            destination=destination,
            places=places
        )

    except ValueError as error:
        return jsonify(
            success=False,
            transcript=transcript,
            message=str(error)
        ), 400

    except requests.exceptions.HTTPError as error:

        app.logger.error(
            "Kakao API HTTP error: %s",
            error
        )

        return jsonify(
            success=False,
            transcript=transcript,
            message=(
                "카카오 장소 검색에 실패했습니다. "
                "카카오맵 API 사용 설정과 키를 확인하세요."
            )
        ), 502

    except Exception:
        app.logger.exception("Navigation failed")

        return jsonify(
            success=False,
            transcript=transcript,
            message="처리 중 오류가 발생했습니다. "
                    "다시 시도해 주세요."
        ), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify(
        success=False,
        message="음성 파일이 너무 큽니다. "
                "짧게 말씀해 주세요."
    ), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )