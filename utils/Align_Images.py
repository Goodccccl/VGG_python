#!usr/bin/python3
# -*-coding:utf-8-*-

from __future__ import print_function
import cv2
import numpy as np

# MAX_MATCHES = 500
MAX_MATCHES = 700
# MAX_MATCHES = 10000
# MAX_MATCHES = 15000
GOOD_MATCH_PERCENT = 0.25


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    # orb =  cv2.xfeatures2SURF_create()
    # orb = cv2.xfeatures2d.SURF_create()
    # orb = cv2.AKAZE_create(MAX_MATCHES)
    # detector = cv2.AKAZE_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # print(matches)

    if not isinstance(matches,list):
        matches=list(matches)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite(r"211013_data\4_test_data_1216X1408\L_image_align\matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    # refFilename = "../success_image/good_img.jpg"
    # refFilename = "../staticimg/base.png"
    refFilename = r"211013_data\4_test_data_1216X1408\L_test/Custom_15_1.png"
    # refFilename = "../staticimg/100_success.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = r"211013_data\4_test_data_1216X1408\L_test/Custom_0_0.png"
    # imFilename = "../cut_labels/cut_image.jpg"
    #

    # def point2area(points, img, color):
    #     """
    #     :param points: 点集合
    #     :param img: 图片位置
    #     :param color: BGR三色
    #     :return:将图片上点包围的区域涂上颜色
    #     """
    #     img = cv2.imread(img)
    #     res = cv2.fillPoly(img, [np.array(points)], color)
    #     cv2.imshow('fillpoly', res)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #
    # if __name__ == '__main__':
    #     points = [(20, 20), (70, 70), (120, 200)]
    #     img = 'lena.png'
    #     color = [255, 255, 255]
    #     point2area(points, img, color)


    # imFilename = "../staticimg/111.jpg"
    # imFilename = "../staticimg/100_success.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    # pointsone = [(350, 0),  (512, 205), (512, 0)]
    # pointstwo = [(0, 0),  (175, 0), (0, 234)]
    # # points = [(20, 20), (70, 70), (120, 200)]
    # color = [255, 0, 0]
    # res = cv2.fillPoly(im, [np.array(pointsone)], color)
    # res = cv2.fillPoly(im, [np.array(pointstwo)], color)
    # pointsthree = [(345, 0), (512, 270), (512, 0)]
    # pointsfour = [(0, 0), (170, 0), (0, 260)]
    # res = cv2.fillPoly(imReference, [np.array(pointsthree)], color)
    # res = cv2.fillPoly(imReference, [np.array(pointsfour)], color)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = r"211013_data\4_test_data_1216X1408\L_image_align\aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
