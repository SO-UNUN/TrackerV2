"""
app.py
SHM Motion Tracker — Web App (Streamlit)
University of Phayao | Faculty of Science | Physics Department
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import tempfile
import os

from modules.video_loader import VideoLoader, get_supported_formats
from modules.marker_tracker import (
    ColorRange, COLOR_PRESETS,
    detect_marker_color_based, get_mask_preview,
    auto_detect_color_from_roi,
    track_video_color, track_video_csrt,
    pixels_to_real_units,
)
from modules.oscillator_analysis import (
    analyze_time_domain, compute_fft,
    fit_damped_oscillator, damped_oscillator,
    calculate_spring_constant,
)
from modules.exporter import (
    df_to_csv_bytes, make_filename, make_metadata_json, build_summary_text,
)

st.set_page_config(page_title="SHM Motion Tracker — UP", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.main-title {
    background: linear-gradient(135deg, #1a2745 0%, #2d4373 100%);
    color: #e6c870; padding: 1.2rem 1.5rem; border-radius: 10px;
    border-left: 4px solid #e6c870; margin-bottom: 1.2rem;
}
.main-title h1 { margin: 0; font-size: 1.8rem; font-weight: 600; }
.main-title p { margin: 0.3rem 0 0 0; color: #b8c5dc; font-size: 0.9rem; }
div[data-testid="metric-container"] {
    background: rgba(230, 200, 112, 0.05);
    border: 1px solid rgba(230, 200, 112, 0.2);
    border-radius: 8px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
  <h1>🌊 SHM Motion Tracker</h1>
  <p>Video-based marker tracking + Simple Harmonic Motion analysis · University of Phayao</p>
</div>
""", unsafe_allow_html=True)

# Session state
ss = st.session_state
for key, default in [
    ("video_path", None), ("video_meta", None), ("first_frame", None),
    ("color_range", None), ("init_bbox", None), ("tracker_method", "color"),
    ("df_processed", None), ("time_result", None),
    ("fft_result", None), ("fit_result", None), ("analysis_done", False),
]:
    if key not in ss:
        ss[key] = default

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio(
        "Analysis Mode",
        ["Core", "Core + Plus", "Advanced"],
        index=0,
        help="Core=tracking+plot | Plus=+period+phase | Advanced=+FFT+damped fit",
    )
    is_plus = mode in ("Core + Plus", "Advanced")
    is_advanced = mode == "Advanced"

    st.divider()
    st.subheader("Tracking Method")
    tracker_choice = st.radio(
        "Algorithm",
        ["Color-based", "CSRT (Object tracker)"],
        index=0,
    )
    ss.tracker_method = "color" if "Color" in tracker_choice else "csrt"

    st.divider()
    st.subheader("Sampling")
    sampling_mode = st.radio(
        "Sampling mode",
        ["Auto (recommended)", "Full (every frame)", "Custom"],
        index=0,
    )
    custom_fps = None
    if sampling_mode == "Custom":
        custom_fps = st.number_input(
            "fps to analyze", min_value=1.0, max_value=240.0, value=30.0, step=1.0,
        )

    st.divider()
    st.subheader("Calibration")
    pixels_per_cm = st.number_input(
        "Pixels per cm", min_value=0.0, value=10.0, step=0.1,
        help="วัดความยาวอ้างอิงใน video แล้วใส่จำนวน pixels per cm",
    )

    if is_advanced:
        st.divider()
        st.subheader("Spring System")
        mass_g = st.number_input("Mass (gram)", min_value=0.0, value=100.0, step=10.0)
        mass_kg = mass_g / 1000.0
    else:
        mass_g = 0
        mass_kg = 0

# ===================== TABS =====================
tab_upload, tab_track, tab_run, tab_results = st.tabs([
    "📤 1. Upload", "🎯 2. Marker Setup",
    "🚀 3. Track", "📊 4. Results",
])

# ===================== TAB 1: UPLOAD =====================
with tab_upload:
    col_up, col_info = st.columns([2, 1])
    with col_up:
        st.subheader("Upload video")
        uploaded = st.file_uploader(
            "เลือกไฟล์วิดีโอ (.mp4, .mov, .avi)",
            type=get_supported_formats(),
        )
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                ss.video_path = tmp.name
            try:
                with VideoLoader(ss.video_path) as loader:
                    ss.video_meta = loader.get_metadata()
                    ss.first_frame = loader.get_first_frame()
                ss.video_meta["filename"] = uploaded.name
                st.success(f"✅ Loaded: {uploaded.name}")
            except Exception as e:
                st.error(f"Cannot load video: {e}")

    with col_info:
        st.subheader("Video info")
        if ss.video_meta:
            m = ss.video_meta
            st.metric("Duration", f"{m['duration_sec']:.1f} sec")
            st.metric("FPS", f"{m['fps']}")
            st.metric("Resolution", f"{m['width']} × {m['height']}")
            st.metric("Total frames", f"{m['total_frames']:,}")
            if sampling_mode == "Auto (recommended)":
                eff_fps = 60 if m['fps'] > 60 else m['fps']
                if m['fps'] > 60:
                    st.info(f"📉 Auto downsample: {m['fps']:.0f} → 60 fps")
                else:
                    st.info(f"✅ ใช้ทุกเฟรม ({m['fps']:.0f} fps)")
            elif sampling_mode == "Full (every frame)":
                eff_fps = m['fps']
            else:
                eff_fps = custom_fps
            est_samples = int(m['duration_sec'] * eff_fps)
            st.metric("Will analyze", f"{est_samples:,} frames")
            nyquist = eff_fps / 2
            practical = eff_fps / 10
            st.caption(
                f"📊 จับความถี่ได้สูงสุด {nyquist:.1f} Hz "
                f"(แม่นยำ ≤ {practical:.1f} Hz)"
            )
            if est_samples > 50_000:
                st.warning(
                    f"⚠️ Video ใหญ่ ({est_samples:,} frames) — "
                    f"ใช้เวลาประมวลผล 5-20 นาที"
                )
        else:
            st.info("⬅ อัปโหลดวิดีโอเพื่อดูข้อมูล")

# ===================== TAB 2: MARKER SETUP =====================
with tab_track:
    if ss.first_frame is None:
        st.info("⬆ อัปโหลดวิดีโอก่อน")
    elif ss.tracker_method == "color":
        st.subheader("🎨 Color-based Marker Setup")

        # === Color picker mode selector ===
        color_mode = st.radio(
            "วิธีเลือกสี marker",
            ["🎯 Pick from ROI (แนะนำ)", "📋 Color Preset", "⚙️ Custom HSV"],
            index=0,
            horizontal=True,
            help="**Pick from ROI** — ลากกรอบรอบ marker → โปรแกรมดูดสีให้\n\n**Preset** — เลือกจากสีมาตรฐาน\n\n**Custom** — ปรับ HSV ด้วยตัวเอง",
        )

        col_setup, col_preview = st.columns([1, 2])

        # =================================================
        # MODE 1: Pick from ROI — ใช้ streamlit-cropper (ลากกรอบ interactive)
        # =================================================
        if "Pick from ROI" in color_mode:
            with col_setup:
                st.write("**📐 ขั้นตอน:**")
                st.write("1. **ลากกรอบสีแดง** บนภาพให้ครอบ marker")
                st.write("2. **ปรับขนาด**โดยลากขอบกรอบ")
                st.write("3. โปรแกรมดูดสีให้อัตโนมัติ")
                st.divider()

                h_tol = st.slider("Hue tolerance (±)", 5, 40, 15,
                                   help="ช่วงสีให้ครอบคลุม — ต่ำ=เข้มงวด, สูง=หลวม")
                s_tol = st.slider("Saturation tolerance (±)", 20, 120, 60)
                v_tol = st.slider("Value tolerance (±)", 20, 120, 60)
                min_area = st.slider("Min marker area (pixels)", 10, 1000, 50)

            with col_preview:
                try:
                    from streamlit_cropper import st_cropper

                    img_h, img_w = ss.first_frame.shape[:2]
                    img_pil = Image.fromarray(ss.first_frame)

                    st.write("**🎯 ลากกรอบแดงให้ครอบ marker:**")

                    # st_cropper จะ resize ภาพให้พอดี container
                    # box_algorithm=None → ผู้ใช้ลากเอง
                    cropped_box = st_cropper(
                        img_pil,
                        realtime_update=True,
                        box_color="#ff4444",
                        aspect_ratio=None,
                        return_type="box",  # คืน {left, top, width, height}
                        stroke_width=3,
                        key="cropper_pick",
                    )

                    if cropped_box is not None:
                        rx = int(cropped_box["left"])
                        ry = int(cropped_box["top"])
                        rw = int(cropped_box["width"])
                        rh = int(cropped_box["height"])

                        # Validate (กันค่า negative หรือ 0)
                        rx = max(0, min(rx, img_w - 10))
                        ry = max(0, min(ry, img_h - 10))
                        rw = max(10, min(rw, img_w - rx))
                        rh = max(10, min(rh, img_h - ry))

                        # ดูดสีจาก ROI
                        ss.color_range = auto_detect_color_from_roi(
                            ss.first_frame, rx, ry, rw, rh,
                            h_tolerance=h_tol, s_tolerance=s_tol, v_tolerance=v_tol,
                        )
                        ss.color_range.label = "picked_from_roi"

                        st.info(
                            f"📌 ROI: ({rx}, {ry}, {rw}×{rh}) | "
                            f"HSV: H=[{ss.color_range.h_low}-{ss.color_range.h_high}], "
                            f"S=[{ss.color_range.s_low}-{ss.color_range.s_high}], "
                            f"V=[{ss.color_range.v_low}-{ss.color_range.v_high}]"
                        )

                        # แสดง preview ของ mask
                        st.write("**Preview (สีเขียวสด = พื้นที่ที่จะถูก track):**")
                        preview = get_mask_preview(ss.first_frame, ss.color_range)
                        st.image(preview, use_column_width=True)

                        pos = detect_marker_color_based(
                            ss.first_frame, ss.color_range, min_area=min_area
                        )
                        if pos:
                            st.success(
                                f"✅ Marker detected at ({pos[0]:.0f}, {pos[1]:.0f}) "
                                f"area={pos[2]:.0f}px"
                            )
                        else:
                            st.warning("⚠️ ไม่พบ marker — ลองเพิ่ม tolerance หรือขยับกรอบ")
                except ImportError:
                    st.error(
                        "📦 ไม่พบ streamlit-cropper — กรุณาเพิ่มใน requirements.txt:\n\n"
                        "`streamlit-cropper>=0.2.0`"
                    )

        # =================================================
        # MODE 2: Color Preset
        # =================================================
        elif "Preset" in color_mode:
            with col_setup:
                preset_name = st.selectbox(
                    "Color Preset",
                    list(COLOR_PRESETS.keys()),
                    index=0,  # Red default
                )
                preset = COLOR_PRESETS[preset_name]
                ss.color_range = ColorRange(
                    h_low=preset.h_low, h_high=preset.h_high,
                    s_low=preset.s_low, s_high=preset.s_high,
                    v_low=preset.v_low, v_high=preset.v_high,
                    label=preset_name,
                )
                min_area = st.slider("Min marker area (pixels)", 10, 1000, 50)
                st.caption(
                    f"HSV: H=[{preset.h_low}-{preset.h_high}], "
                    f"S=[{preset.s_low}-{preset.s_high}], "
                    f"V=[{preset.v_low}-{preset.v_high}]"
                )

            with col_preview:
                st.write("**Preview: พื้นที่สีเขียวสด = ที่จะถูก track**")
                preview = get_mask_preview(ss.first_frame, ss.color_range)
                st.image(preview, use_column_width=True)
                pos = detect_marker_color_based(
                    ss.first_frame, ss.color_range, min_area=min_area
                )
                if pos:
                    st.success(
                        f"✅ Marker detected at ({pos[0]:.0f}, {pos[1]:.0f}) "
                        f"area={pos[2]:.0f}px"
                    )
                else:
                    st.warning("⚠️ ไม่พบ marker — ลอง preset อื่นหรือเปลี่ยนเป็น Pick from ROI")

        # =================================================
        # MODE 3: Custom HSV sliders
        # =================================================
        else:  # Custom
            with col_setup:
                st.caption("ปรับ HSV range ด้วยตัวเอง — เหมาะสำหรับผู้ที่เข้าใจ HSV")
                h_range = st.slider("Hue (0-179)", 0, 179, (40, 80))
                s_range = st.slider("Saturation (0-255)", 0, 255, (100, 255))
                v_range = st.slider("Value (0-255)", 0, 255, (50, 255))
                ss.color_range = ColorRange(
                    h_low=h_range[0], h_high=h_range[1],
                    s_low=s_range[0], s_high=s_range[1],
                    v_low=v_range[0], v_high=v_range[1],
                    label="custom",
                )
                min_area = st.slider("Min marker area (pixels)", 10, 1000, 50)

            with col_preview:
                st.write("**Preview: พื้นที่สีเขียวสด = ที่จะถูก track**")
                preview = get_mask_preview(ss.first_frame, ss.color_range)
                st.image(preview, use_column_width=True)
                pos = detect_marker_color_based(
                    ss.first_frame, ss.color_range, min_area=min_area
                )
                if pos:
                    st.success(
                        f"✅ Marker detected at ({pos[0]:.0f}, {pos[1]:.0f}) "
                        f"area={pos[2]:.0f}px"
                    )
                else:
                    st.warning("⚠️ ไม่พบ marker — ลองปรับ HSV range หรือลด min area")

    else:
        st.subheader("🎯 CSRT Object Tracking Setup")
        st.caption("ลากกรอบรอบ marker บนภาพ — ปรับขนาดให้พอดี")

        img_h, img_w = ss.first_frame.shape[:2]

        try:
            from streamlit_cropper import st_cropper

            img_pil = Image.fromarray(ss.first_frame)
            cropped_box = st_cropper(
                img_pil,
                realtime_update=True,
                box_color="#12d022",
                aspect_ratio=None,
                return_type="box",
                stroke_width=3,
                key="cropper_csrt",
            )

            if cropped_box is not None:
                bx = int(cropped_box["left"])
                by = int(cropped_box["top"])
                bw = int(cropped_box["width"])
                bh = int(cropped_box["height"])

                bx = max(0, min(bx, img_w - 10))
                by = max(0, min(by, img_h - 10))
                bw = max(10, min(bw, img_w - bx))
                bh = max(10, min(bh, img_h - by))

                ss.init_bbox = (bx, by, bw, bh)
                st.success(f"✅ Bounding box: ({bx}, {by}, {bw}×{bh})")
        except ImportError:
            st.error(
                "📦 ไม่พบ streamlit-cropper — กรุณาเพิ่มใน requirements.txt:\n\n"
                "`streamlit-cropper>=0.2.0`"
            )


# ===================== TAB 3: TRACK =====================
with tab_run:
    if ss.video_path is None:
        st.info("⬆ Upload video first")
    elif ss.tracker_method == "color" and ss.color_range is None:
        st.info("⬆ Setup color range first")
    elif ss.tracker_method == "csrt" and ss.init_bbox is None:
        st.info("⬆ Draw bounding box first")
    else:
        st.subheader("🚀 Run Tracking")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.write(f"**Method:** {ss.tracker_method.upper()}")
            st.write(f"**Mode:** {mode}")
            if ss.tracker_method == "color":
                st.write(f"**Color:** {ss.color_range.label}")
            else:
                st.write(f"**BBox:** {ss.init_bbox}")
        with col_b:
            run_btn = st.button("🚀 START TRACKING", type="primary", use_container_width=True)

        if run_btn:
            fps = ss.video_meta['fps']
            if sampling_mode == "Auto (recommended)":
                use_fps = 60.0 if fps > 60 else None
            elif sampling_mode == "Full (every frame)":
                use_fps = None
            else:
                use_fps = custom_fps

            progress = st.progress(0, text="Tracking marker...")
            status = st.empty()

            with VideoLoader(ss.video_path) as loader:
                total_est = loader.estimate_total_samples(use_fps)

                def cb(current, total):
                    if current % 50 == 0 and total > 0:
                        pct = min(1.0, current / total)
                        progress.progress(pct, text=f"Frame {current}/{total}")

                if ss.tracker_method == "color":
                    df = track_video_color(
                        loader.iter_frames(sampling_fps=use_fps),
                        ss.color_range, min_area=50,
                        progress_callback=cb, total_estimate=total_est,
                    )
                else:
                    df = track_video_csrt(
                        loader.iter_frames(sampling_fps=use_fps),
                        ss.init_bbox,
                        progress_callback=cb, total_estimate=total_est,
                    )
            progress.progress(1.0, text="✅ Done!")

            if pixels_per_cm > 0:
                df = pixels_to_real_units(df, pixels_per_cm)
            ss.df_processed = df

            n_found = df["found"].sum() if "found" in df.columns else len(df)
            status.success(
                f"✅ Tracked {len(df)} frames ({n_found} successful, "
                f"{100*n_found/len(df):.1f}%)"
            )

            # Reset analysis results (จะคำนวณใหม่ใน Tab 4 ตาม axis ที่เลือก)
            ss.time_result = None
            ss.fft_result = None
            ss.fit_result = None
            ss.analysis_done = True
            st.rerun()

        if ss.analysis_done and ss.df_processed is not None:
            st.divider()
            st.success("✅ Tracking complete — go to **Results** tab")
            st.dataframe(ss.df_processed.head(10), use_container_width=True)

# ===================== TAB 4: RESULTS =====================
with tab_results:
    if not ss.analysis_done or ss.df_processed is None:
        st.info("⬅ Run tracking first")
    else:
        df = ss.df_processed
        x_col = "x_cm" if "x_cm" in df.columns else "x_px"
        y_col = "y_cm" if "y_cm" in df.columns else "y_px"
        unit = "cm" if "x_cm" in df.columns else "px"

        # =====================================================
        # AXIS SELECTOR — ผู้ใช้เลือกแกนสำหรับวิเคราะห์
        # =====================================================
        st.markdown("### 🎯 Select axis for analysis")
        axis_choice = st.radio(
            "ใช้ข้อมูลแกนใดในการวิเคราะห์ (peak, FFT, damped fit):",
            ["Y axis (สำหรับสปริงสั่นแนวดิ่ง)", "X axis (สำหรับลูกตุ้มแกว่ง)"],
            index=0,
            horizontal=True,
            help="เลือกตามทิศทางการสั่นของวัตถุ\n• สปริง = สั่นแนวดิ่ง = Y axis\n• ลูกตุ้ม = แกว่งซ้าย-ขวา = X axis",
        )
        analysis_axis = "y" if "Y axis" in axis_choice else "x"
        analysis_col = y_col if analysis_axis == "y" else x_col
        axis_label = "Y" if analysis_axis == "y" else "X"

        # คำนวณ analysis ตามแกนที่เลือก (recompute เมื่อเปลี่ยน)
        ss.time_result = analyze_time_domain(
            df["time_sec"].values, df[analysis_col].values
        )
        if is_advanced:
            ss.fft_result = compute_fft(
                df["time_sec"].values, df[analysis_col].values
            )
            ss.fit_result = fit_damped_oscillator(
                df["time_sec"].values, df[analysis_col].values
            )

        st.divider()

        # Position plots
        st.subheader(f"📈 Position vs Time")
        fig_pos = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=(
                f"X position ({unit})" + (" ← analyzed" if analysis_axis == "x" else ""),
                f"Y position ({unit})" + (" ← analyzed" if analysis_axis == "y" else ""),
            ),
            vertical_spacing=0.08,
        )
        # X axis trace
        fig_pos.add_trace(go.Scatter(
            x=df["time_sec"], y=df[x_col],
            name="x", line=dict(
                color="#3b82f6",
                width=2.5 if analysis_axis == "x" else 1.2,
            ),
        ), row=1, col=1)
        # Y axis trace
        fig_pos.add_trace(go.Scatter(
            x=df["time_sec"], y=df[y_col],
            name="y", line=dict(
                color="#ef4444",
                width=2.5 if analysis_axis == "y" else 1.2,
            ),
        ), row=2, col=1)
        # Peak markers — แสดงบนแกนที่วิเคราะห์
        if is_plus and ss.time_result and ss.time_result.n_peaks > 0:
            peak_row = 1 if analysis_axis == "x" else 2
            fig_pos.add_trace(go.Scatter(
                x=ss.time_result.peak_times, y=ss.time_result.peak_values,
                mode="markers", name="peaks",
                marker=dict(color="#e6c870", size=10, symbol="circle-open",
                            line=dict(width=2)),
            ), row=peak_row, col=1)
        fig_pos.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_pos.update_layout(height=500, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_pos, use_container_width=True)

        # Time Domain
        if ss.time_result:
            tr = ss.time_result
            st.subheader(f"📐 Time Domain Analysis ({axis_label} axis)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period T", f"{tr.period_sec:.4f} s")
            c2.metric("Frequency f", f"{tr.frequency_hz:.4f} Hz")
            c3.metric("Amplitude", f"{tr.amplitude:.3f} {unit}")
            c4.metric("Peaks found", tr.n_peaks)

        # Phase space (Plus) — ใช้แกนที่เลือก
        if is_plus:
            st.divider()
            st.subheader(f"🌀 Phase Space ({axis_label} axis vs velocity)")
            t = df["time_sec"].values
            x = df[analysis_col].values
            v = np.gradient(x, t)
            fig_ps = go.Figure()
            fig_ps.add_trace(go.Scatter(
                x=x, y=v, mode="markers",
                marker=dict(color=t, colorscale="Viridis", size=3,
                            showscale=True, colorbar=dict(title="Time (s)")),
            ))
            fig_ps.update_layout(
                xaxis_title=f"{axis_label} position ({unit})",
                yaxis_title=f"{axis_label} velocity ({unit}/s)",
                template="plotly_white", height=450,
            )
            st.plotly_chart(fig_ps, use_container_width=True)

        # FFT (Advanced)
        if is_advanced and ss.fft_result:
            st.divider()
            st.subheader(f"📊 FFT Spectrum ({axis_label} axis)")
            fft = ss.fft_result
            mask = fft.frequencies > 0.1
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=fft.frequencies[mask], y=fft.amplitudes[mask],
                mode="lines", line=dict(color="#22c55e", width=2),
                fill="tozeroy", fillcolor="rgba(34, 197, 94, 0.2)",
                name="FFT",
            ))
            fig_fft.add_vline(
                x=fft.dominant_freq,
                line=dict(color="#e6c870", dash="dash", width=2),
                annotation_text=f"f = {fft.dominant_freq:.3f} Hz",
                annotation_position="top",
            )
            fig_fft.update_layout(
                xaxis_title="Frequency (Hz)", yaxis_title="Amplitude",
                template="plotly_white", height=400,
                xaxis=dict(range=[0, min(fft.sample_rate/2, 20)]),
            )
            st.plotly_chart(fig_fft, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Dominant freq", f"{fft.dominant_freq:.4f} Hz")
            c2.metric("Dominant period", f"{fft.dominant_period:.4f} s")
            c3.metric("Sample rate", f"{fft.sample_rate:.1f} Hz")

            if ss.time_result and ss.time_result.frequency_hz > 0:
                diff_pct = abs(fft.dominant_freq - ss.time_result.frequency_hz) / ss.time_result.frequency_hz * 100
                if diff_pct < 5:
                    st.success(
                        f"✅ Time ({ss.time_result.frequency_hz:.4f}) vs "
                        f"FFT ({fft.dominant_freq:.4f}) Hz สอดคล้องกัน "
                        f"(diff {diff_pct:.1f}%)"
                    )

        # Damped fit (Advanced) — ใช้แกนที่เลือก
        if is_advanced and ss.fit_result and ss.fit_result.success:
            st.divider()
            st.subheader(f"🌊 Damped Oscillator Fit ({axis_label} axis)")
            fr = ss.fit_result
            t = df["time_sec"].values
            x_data = df[analysis_col].values
            x_fit = damped_oscillator(t, fr.A, fr.gamma, fr.omega, fr.phi, fr.offset)
            env_pos = fr.A * np.exp(-fr.gamma * t) + fr.offset
            env_neg = -fr.A * np.exp(-fr.gamma * t) + fr.offset

            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(
                x=t, y=x_data, mode="markers",
                line=dict(color="#9eb0d0", width=1), name="Data",
            ))
            fig_fit.add_trace(go.Scatter(
                x=t, y=x_fit, mode="lines",
                line=dict(color="#ef4444", width=2), name="Fit",
            ))
            fig_fit.add_trace(go.Scatter(
                x=t, y=env_pos, mode="lines",
                line=dict(color="#eee302", dash="dash", width=1.5), name="Envelope",
            ))
            fig_fit.add_trace(go.Scatter(
                x=t, y=env_neg, mode="lines",
                line=dict(color="#eee302", dash="dash", width=1.5), showlegend=False,
            ))
            fig_fit.update_layout(
                xaxis_title="Time (s)", yaxis_title=f"Position ({unit})",
                template="plotly_white", height=420, hovermode="x unified",
            )
            st.plotly_chart(fig_fit, use_container_width=True)

            st.code(
                f"x(t) = {fr.A:.3f}·exp(-{fr.gamma:.4f}·t)·cos({fr.omega:.3f}·t + {fr.phi:.3f}) + {fr.offset:.3f}",
                language="text",
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Damping ratio ζ", f"{fr.damping_ratio:.4f}")
            c2.metric("Quality factor Q", f"{fr.quality_factor:.2f}")
            c3.metric("Natural freq f₀", f"{fr.natural_freq_hz:.4f} Hz")
            c4.metric("R²", f"{fr.r_squared:.4f}")

            if mass_kg > 0:
                k_exp = calculate_spring_constant(mass_kg, fr.natural_freq_hz)
                st.divider()
                st.subheader("🔬 Spring System")
                c1, c2 = st.columns(2)
                c1.metric("Spring constant k", f"{k_exp:.3f} N/m",
                          help="k = (2πf₀)²·m")
                c2.metric("Mass m", f"{mass_g:.1f} g")

        # Export
        st.divider()
        st.subheader("💾 Export")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 Download tracking data (CSV)",
                data=df_to_csv_bytes(df),
                file_name=make_filename("shm_tracking", "csv"),
                mime="text/csv", use_container_width=True,
            )
        with c2:
            meta = make_metadata_json(
                video_meta=ss.video_meta,
                tracking_info={
                    "method": ss.tracker_method,
                    "pixels_per_cm": pixels_per_cm,
                },
                analysis_results={
                    "time_domain": ss.time_result.to_dict() if ss.time_result else None,
                    "fft_dominant_freq": ss.fft_result.dominant_freq if ss.fft_result else None,
                    "damped_fit": {
                        "natural_freq_hz": ss.fit_result.natural_freq_hz,
                        "damping_ratio": ss.fit_result.damping_ratio,
                        "Q": ss.fit_result.quality_factor,
                        "R_squared": ss.fit_result.r_squared,
                    } if ss.fit_result and ss.fit_result.success else None,
                },
                settings={"mode": mode, "tracker": ss.tracker_method},
            )
            st.download_button(
                "📥 Download analysis (JSON)",
                data=meta,
                file_name=make_filename("shm_analysis", "json"),
                mime="application/json", use_container_width=True,
            )

        with st.expander("📄 Summary"):
            st.text(build_summary_text(
                ss.video_meta, df,
                ss.time_result, ss.fft_result, ss.fit_result,
            ))

st.divider()
st.caption(
    "🌊 **SHM Motion Tracker** — Video-based SHM analysis. "
    "Developed by Physics Department, University of Phayao."
)
