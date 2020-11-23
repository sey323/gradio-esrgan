import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")


def predict(
    inp,
):
    """
    ESRGANを用いて、入力画像を4倍の解像度に変換する。
    https://tfhub.dev/captain-pool/esrgan-tf2/1
    inp :
        入力画像
    """
    h, w, c = inp.shape  # 入力画像から画像サイズを取得する
    inp = inp.reshape((-1, h, w, c))
    inp = tf.cast(inp, tf.float32)
    output = model(inp)
    output = tf.cast(tf.clip_by_value(output[0], 0, 255), tf.uint8)
    return output


def gradio_gui():
    """
    gradioのGui画面を定義する
    """
    image = gr.inputs.Image(label="Input Image",)
    output = gr.outputs.Image(label="Output Image", type="numpy")

    interface = gr.Interface(fn=predict, inputs=image, outputs=output)
    interface.launch()


if __name__ == "__main__":
    gradio_gui()
