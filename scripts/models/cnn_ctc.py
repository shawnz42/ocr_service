#!/usr/bin/env python
# -*- coding: utf-8 -*-  


"""
@desc: 
@author: 
@time: 17-12-18 下午11:33
"""


import tensorflow as tf
import numpy as np


from scripts.settings import MODELS_PATH
from scripts.models.model_1 import (graph, saver,  x, keep_prob, seq_len,decoded, MAX_WIDE_SHRINK)
from scripts.utils.image_tools import str2gray
from scripts.data.hanzi_handler import hanzi_handler




def main(img_bs64):
    """

    :param img_bs64:
    :return:
    """

    im = str2gray(img_bs64)
    assert im.shape[0] == 17
    im = im[1:, :]
    im = 1 - im

    max_wide = im.shape[1]

    img_extend_with_channel = np.expand_dims(im, axis=2)
    input = np.expand_dims(img_extend_with_channel, axis=0)

    val_feed = {x: input,
                keep_prob: 1,
                seq_len: [max_wide / MAX_WIDE_SHRINK]  # need compute
                }

    with tf.Session(graph=graph) as sess:
        checkpoint = tf.train.latest_checkpoint(MODELS_PATH)
        if checkpoint:
            saver.restore(sess, checkpoint)
        else:
            print("no model")


        train_decoded = sess.run(decoded[0], feed_dict=val_feed)

        decoded_real_list = hanzi_handler.decode_sparse_tensor(train_decoded)


        return decoded_real_list[0]


if __name__ == "__main__":
    # img_bs64 = "iVBORw0KGgoAAAANSUhEUgAAAQAAAAARBAMAAADAoW/fAAAAG1BMVEX///8AAP9/f/+/v/8fH/+fn/9fX/8/P//f3/+rxP93AAAD1UlEQVRIic2Vy3PbNhCHlwQfOIKh6PCoxJ44Rzq1Ex1l62EdydrT+sg4Sqwjk8iuj6DE2Pizu7sgRU1m2pnmVGgGHOGx++G3uwDA/6cJ/HEz/75O/rIH/Q973SFABlBnjub/d9R5rTdDDfcqDQ0uzQF41faQpyNqvEvqbrGWaE+qW+xia2TVOlLg0OcTntHonXsv88cgcOkRmpYIAg1NOmxcOdaFHKuU/opLzQDR5S17TsECuWO3AjjhxSms8ajqKTObBdP7FVoZWNgoAngDklZ1LagghrAEmKGpsDpvlznIi8wOOcHue/yo0CpsaprNUlc72ukBmrw7LeBYwCTx19aUh6eSFzgzJl0DYlTueAeAO79AocFHgfSaDl8DK5B/c8u8A/j84i0BSJx0tJelwaImPbsQYNSONDC8SinQkvCvbVZ9CTJohjsFruB3qby8T4ElLkS+BgE+LKAFcKPo+dH8pTsAUJKMF5kFSNYPd/oaOgV8/IaLToFgIpkErdJOMS60P7BqkfFruJfqdC8Hn/VWYACXCDDFCEd2swMPaKpsAShopIAbM8BvgzyIo2SnQIHnkeusC8FHo1AsJSx6ECVifTbXVTOdThchhNuxVPe9f2ObSBScPOJ5jDkxxuYrMrdJCC4aJF3fWgWG4LM7t6TefwFNBs8xi4S7hGSAcGXqEaZ7sJoO4SlmZVO2GKuHHqANjb5RTxRNZK7Jk8MyJs7IpKbWEA5GbB1sEjpWJig47UXOpZtbRsS2IfBaMa/1CKjQa1TAxgHhhnsxsPeDTF/W+wAApwOuORbyQ7JQb+AQDq0COELu3IWXvyaPWMaaM48BGhuCtkLEwTFRZxDiQMUnNkqo3r9M4IQKRZHZ2x3AOzlJqh3AvTKq+TGDWQfAHpb6CCYE4A1F2gMUNgRwqp3tI4jbK4C+7MmeUfsDTQluzLvR0TcGEMo/nz8P07sOQIzRoHupheoA/BI3aDd1KzrqMRUTdDlwbEMgL9rCFlwCcoAhOGd7Jn63dw9cYvcqZ4BcJgQQRqV/I+aQhpPuJswR4OxBB6UFCOF0PqpATLMC4bgM8RysOjq4sArgiKOD2F/OLUAbEwI4GPT3QFNRNic0j8Vf+Zhn0tgET2HlUIVQbijxchZceLlNQvnn5P30ExQrNzZPFRSD82gQUzaB2aYyRvfpD8z0EV5wH7c3LEjaKcA5sJeCf3D/GfgJ4Yew6aMFAdrkTrkzgDMfNlR2mwjr2P+aSVhmEabq5mqk7RsGr6JSYFTk9+kwg+hg7/HsFMALWfYB+LXGbxab1v9hF6nu/zz4N9Qu3wZWiUCoAAAAAElFTkSuQmCC"
    img_bs64 = "iVBORw0KGgoAAAANSUhEUgAAAPAAAAARBAMAAAARYzyGAAAAG1BMVEX///8AAP/f3/8fH/+/v/+fn/9/f/9fX/8/P/+NOBI5AAADtElEQVRIic2Vz3faOBDHv5J/Hi2HJBwlO7+OxgZytUloOZqXfW2PTuju9mhI9jVHIK+8/tk7kqBJoYftbfUICpI8n/nOjMbA/2B48uff8leHfnNw/fWA8a83I03NcbdbuDJIL9PffmmdcCRimiMkSUIz7bELmoN6z5jeTMwAj91aoYYbo9IWgxLcPPwTGPcjQSOeixYbDyiK4qgoMijjhHv8A0yHzAyW1jxRI0JwISq7PMMXOizMkH7rHisSFB4VvaJYwm+haHkf7Kat+eG1YeOvybKEpJ2VsYFpek9zRMsx1BbMZc8ieKTIP1rmA6hya5VJR8R3Ki0xh3lkTicVubYfanzXDDC/WRW9rpyR4KNh4+fGBvi9USxER4NJU8/4wXNr4G9mFAckK4djvGHyIwVdnUmf5NKn9kqvURi8BdOxLMxl3wdDGjTjEQ+MKHLoRusl8F3PgN2tYq/DmJM0lkGB6TOjeEpxfJE7xbl2VER+ZRV7WVgrN3sLjpNE4jvicMkwCeqrJlwj1YoLnMRkgclg4FB6RO3HFhyeEvi8xcoQeBwbsHesK7O/A6diGWn/mVWM9Q1U+LYYbY45TRSKSXh7PcInuVXcWDCHf9nQIafbGvCEFIeTMuxZ8DbU04b3wVelMjmT1gKBrWIdnCpsD8BmQ8RTcfIHTZ52Ule1BiswW0XRx+jJgPvHQpzO8FJtQ63YBTk91kmmalGaxKj0YlMIVrGpBoXNAfiDtzxHCozztKFMnBO292Tckdpf/pJhhDyiwxTTZeQw/wQLtVXsVGeZT+mrSm7MG/BOsb+NPelRqA7AX/02bVLMTjqD3Gn8foSWXeakmCfPLnDXbiYDNBH5pLwHGaXMy0Z4Bctv95S+oWkVe2BHx67QC7FZ3gMPgobXKf6Scf3lWRc1zyg5BA5F/wafBFJdOhEFU1EnoKym2ZNuCDrUnamSak1tbomAdL/m2IbaBtmAb1/vcZKcxbr7xIHJonZRdWi9vepSF4Km4dkZKvfz42faVDXUVQO6QJtv+U7x6ZAl4765TvzalpLhLHbF9QM8f73HtmyE7sw8+Qp+nl105wu8exy6s+sSumGJPkdVLNz5EtFK0uWVOF4yWWUaMZrqBsce6+nCEwV1llK/GXhSSffoIeJnprioXrY5PhgB1XkgdNv50K3dVfme5OGitB2f/r0szbHoH5jXVkR/rmkaLxudK7WAu+Z/3iamsCno4pTuIiKfWn+oF0I49GpJD8FvUi63b7P/PprffQD4FwjNl2k4QTGmAAAAAElFTkSuQmCC"
    ans = main(img_bs64)
    print(ans)
