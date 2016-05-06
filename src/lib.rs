extern crate array_cuda;
extern crate array_new;
extern crate epeg;
extern crate rembrandt_kernels;
extern crate turbojpeg;
extern crate varraydb;
//extern crate vips;

extern crate byteorder;
//extern crate chan;
extern crate csv;
extern crate image;
//extern crate magick;
extern crate memmap;
extern crate sdl2;
extern crate sdl2_image;
extern crate stb_image;
extern crate tar;
extern crate threadpool;

extern crate rand;
extern crate rustc_serialize;
extern crate time;

use array_cuda::device::context::{DeviceContext};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer};
use array_new::{ArrayZeroExt, NdArraySerialize, Array3d};
use rembrandt_kernels::ffi::*;
use turbojpeg::{TurbojpegDecoder, TurbojpegEncoder};
use varraydb::{VarrayDb};

use byteorder::{WriteBytesExt, LittleEndian};
use csv::{Reader as CsvReader};
use epeg::{EpegImage};
use image::{GenericImage, ImageBuffer, Rgb, ColorType, ImageDecoder, DecodingResult};
use image::{DynamicImage, ImageFormat, load};
use image::imageops::{FilterType, resize};
//use image::jpeg::{JPEGDecoder};
//use magick::{MagickWand, FilterType as MagickFilterType};
use memmap::{Mmap, Protection};
use sdl2::rect::{Rect};
use sdl2::pixels::{PixelFormatEnum};
use sdl2::rwops::{RWops};
use sdl2::surface::{Surface};
use sdl2_image::{ImageRWops, SaveSurface};
use stb_image::image::{Image, LoadResult, load_from_memory};
use tar::{Archive};
use threadpool::{ThreadPool};
//use vips::{Vips, VipsImageFormat, VipsImage};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, create_dir_all};
use std::io::{Read, BufRead, Seek, Write, BufReader, Cursor};
use std::iter::{repeat};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread::{spawn};
use time::{get_time};

const NUM_ENCODERS: usize = 6;

#[derive(Clone, Debug)]
pub struct IlsvrcConfig {
  pub wnid_to_rawcats_path: PathBuf,
  pub train_archive_path:   PathBuf,
  pub valid_archive_path:   PathBuf,
  pub valid_rawcats_path:   PathBuf,

  pub resize_smaller_dim:   Option<u32>,
  pub keep_aspect_ratio:    bool,

  pub train_data_path:      PathBuf,
  pub train_labels_path:    PathBuf,
  pub valid_data_path:      PathBuf,
  pub valid_labels_path:    PathBuf,
}

#[derive(RustcDecodable)]
struct WnidRawCatRecord {
  wnid: String,
  raw_cat: i32,
}

enum EncoderMsg {
  RawImageFile(usize, i32, Vec<u8>),
  Quit,
}

enum WriterMsg {
  Skip(usize),
  ImageFile(usize, i32, Vec<u8>),
  Quit,
}

struct ReaderWorker {
  config:       IlsvrcConfig,
  encoder_txs:  Vec<SyncSender<EncoderMsg>>,
  wnid_to_rawcats:  HashMap<String, i32>,
  //archive_file:     File,
  //archive_map:      Mmap,
}

impl ReaderWorker {
  pub fn run(&mut self, archive_map: Mmap) {
    let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
    let mut train_archive = Archive::new(reader);
    let mut counter = 0;
    let start_time = get_time();
    for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
      let wnid_file = wnid_file.unwrap();
      let wnid_path: PathBuf = wnid_file.header().path().unwrap().into_owned();
      let mut wnid_archive = Archive::new(wnid_file);
      //println!("DEBUG: processing {} {:?}...", wnid_idx, wnid_path);
      for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
        let mut im_file = im_file.unwrap();

        let im_path = im_file.header().path().unwrap().into_owned();
        //let im_stem = im_path.file_stem().unwrap().to_str().unwrap();
        let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
        let im_stem_toks: Vec<_> = im_path_toks[0].splitn(2, "_").collect();
        if !self.wnid_to_rawcats.contains_key(im_stem_toks[0]) {
          println!("WARNING: unknown key: {} {} {:?} {:?}", wnid_idx, im_idx, im_path, im_stem_toks);
          continue;
        }
        let rawcat = *self.wnid_to_rawcats.get(im_stem_toks[0]).unwrap();

        let mut buf: Vec<u8> = vec![];
        match im_file.read_to_end(&mut buf) {
          Err(e) => panic!("failed to read image: {:?}", e),
          Ok(_) => {}
        }

        self.encoder_txs[counter % NUM_ENCODERS].send(
            EncoderMsg::RawImageFile(counter, rawcat, buf),
        ).unwrap();
        counter += 1;
      }
      let lap_time = get_time();
      let elapsed = (lap_time - start_time).num_milliseconds() as f32 * 0.001;
      println!("DEBUG: processed: {} elapsed: {:.3}", counter, elapsed);
    }
    for enc_idx in 0 .. NUM_ENCODERS {
      self.encoder_txs[enc_idx].send(
          EncoderMsg::Quit,
      ).unwrap();
    }
  }
}

struct ValidReaderWorker {
  config:       IlsvrcConfig,
  encoder_txs:  Vec<SyncSender<EncoderMsg>>,
  valid_rawcats:    Vec<i32>,
  //wnid_to_rawcats:  HashMap<String, i32>,
  //archive_file:     File,
  //archive_map:      Mmap,
}

impl ValidReaderWorker {
  pub fn run(&mut self, archive_map: Mmap) {
    let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
    let mut archive = Archive::new(reader);
    let mut counter = 0;
    let start_time = get_time();
    for (k, im_file) in archive.entries().unwrap().enumerate() {
      let mut im_file = im_file.unwrap();

      let im_path = im_file.header().path().unwrap().into_owned();
      let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
      let im_stem_toks: Vec<_> = im_path_toks[0].splitn(3, "_").collect();
      let idx: usize = im_stem_toks[2].parse().unwrap();
      assert!(idx >= 1);
      assert!(idx <= 50_000);

      let rawcat = self.valid_rawcats[idx-1];
      let label_cat = rawcat - 1;
      assert!(label_cat >= 0 && label_cat < 1000);
      let mut label_buf = vec![];
      label_buf.write_i32::<LittleEndian>(label_cat).unwrap();
      assert_eq!(4, label_buf.len());

      let mut buf: Vec<u8> = vec![];
      match im_file.read_to_end(&mut buf) {
        Err(e) => panic!("failed to read image: {} {:?}", k, e),
        Ok(_) => {}
      }

      self.encoder_txs[counter % NUM_ENCODERS].send(
          EncoderMsg::RawImageFile(counter, rawcat, buf),
      ).unwrap();
      counter += 1;

      if (k+1) % 100 == 0 {
        let lap_time = get_time();
        let elapsed = (lap_time - start_time).num_milliseconds() as f32 * 0.001;
        println!("DEBUG: processed: {} elapsed: {:.3}", counter, elapsed);
      }
    }
    for enc_idx in 0 .. NUM_ENCODERS {
      self.encoder_txs[enc_idx].send(
          EncoderMsg::Quit,
      ).unwrap();
    }
  }
}

/*struct MagickEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl MagickEncoderWorker {
  pub fn run(&mut self) {
    let wand = MagickWand::new();
    loop {
      match self.encoder_rx.recv() {
        //Err(_) | Ok(EncoderMsg::Quit) => {
        Err(_) => {
          break;
        }
        Ok(EncoderMsg::RawImageFile(idx, rawcat, raw_buf)) => {
          //let mut reader = Cursor::new(&buf);

          match wand.read_image_blob(raw_buf) {
            Err(e) => {
              println!("WARNING: decoding failed: {} {:?}", idx, e);
              continue;
            }
            Ok(_) => {}
          }

          let image_width = wand.get_image_width();
          let image_height = wand.get_image_height();
          let min_side = min(image_width, image_height);

          if min_side <= self.config.resize_smaller_dim {
            // Do nothing.
          } else {
            wand.resize_image(image_width, image_height, MagickFilterType::LanczosFilter, 1.0);
          }

          match wand.write_image_blob("PNG") {
            Err(e) => {
              println!("WARNING: failed to save as png: {} {:?}", idx, e);
              continue;
            }
            Ok(encoded_buf) => {
              let test_buf = encoded_buf.clone();
              match wand.read_image_blob(test_buf) {
                Err(e) => {
                  println!("WARNING: failed to correctly save as png: {} {:?}", idx, e);
                  continue;
                }
                Ok(_) => {}
              }

              self.writer_tx.send(
                  WriterMsg::ImageFile(idx, rawcat, encoded_buf),
              ).unwrap();
            }
          }
        }
      }
    }
  }
}*/

struct ValidTurboEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl ValidTurboEncoderWorker {
  pub fn run(&mut self, dev_idx: usize) {
    let context = DeviceContext::new(dev_idx);
    let ctx = &context.as_ref();
    // XXX(20160505): this takes about 9 GB; yes, some images are large.
    let mut resize_src_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);
    let mut resize_dst_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);

    let mut decoder = TurbojpegDecoder::create().unwrap();
    let mut encoder = TurbojpegEncoder::create().unwrap();

    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          let top_10k = if idx < 10000 {
            0
          } else if idx < 20000 {
            1
          } else if idx < 30000 {
            2
          } else if idx < 40000 {
            3
          } else if idx < 50000 {
            4
          } else {
            unreachable!();
          };

          let image_arr = match decoder.decode_rgb8(&buf) {
            Err(e) => {
              println!("WARNING: turbo decoder failed: {}", idx);
              let im_arr = Array3d::<u8>::zeros((256, 256, 3));
              im_arr
            }

            Ok((header, data)) => {
              let mut im = Image::new(header.width, header.height, 3, data);

              //assert_eq!(3, im.depth);
              if im.depth != 3 && im.depth != 1 {
                panic!("WARNING: stb loaded an unsupported depth: {} {}", idx, im.depth);
              }
              assert_eq!(im.depth * im.width * im.height, im.data.len());

              if im.depth == 1 {
                let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                assert_eq!(im.width * im.height, im.data.len());
                for i in 0 .. im.data.len() {
                  rgb_data.push(im.data[i]);
                  rgb_data.push(im.data[i]);
                  rgb_data.push(im.data[i]);
                }
                assert_eq!(3 * im.width * im.height, rgb_data.len());
                im = Image::new(im.width, im.height, 3, rgb_data);
              }
              assert_eq!(3, im.depth);

              {
                let buf = match im.write_png() {
                  Err(e) => panic!("stb failed to write png: {} {:?}", idx, e),
                  Ok(buf) => buf,
                };
                let orig_path = PathBuf::from(&format!("tmp/preproc_debug/{}/valid_{}_orig.png", top_10k, idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();
              }

              let orig_w = im.width;
              let orig_h = im.height;
              let min_dim = min(orig_w, orig_h);
              if min_dim != 256 {
                let (new_w, new_h) = if min_dim == orig_w {
                  (256, (256 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
                } else {
                  ((256 as f32 / orig_h as f32 * orig_w as f32).round() as usize, 256)
                };

                let mut src_buf_h = Vec::with_capacity(3 * im.width * im.height);
                for c in 0 .. 3 {
                  for y in 0 .. im.height {
                    for x in 0 .. im.width {
                      src_buf_h.push(im.data[c + x * 3 + y * 3 * im.width] as f32 / 255.0);
                    }
                  }
                }

                resize_src_buf.as_ref_mut_range(0, orig_w * orig_h * 3, ctx)
                  .sync_load(&src_buf_h);

                let mut curr_w = orig_w;
                let mut curr_h = orig_h;
                while (curr_w+1)/2 >= new_w && (curr_h+1)/2 >= new_h {
                  unsafe { rembrandt_kernel_image3_bilinear_scale(
                      resize_src_buf.as_ref(ctx).as_ptr(),
                      curr_w as i32, curr_h as i32, 3,
                      resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                      ((curr_w+1)/2) as i32, ((curr_h+1)/2) as i32,
                      ctx.stream.ptr,
                  ) };
                  resize_dst_buf.as_ref(ctx).send(&mut resize_src_buf.as_ref_mut(ctx));
                  curr_w = (curr_w+1)/2;
                  curr_h = (curr_h+1)/2;
                }
                if curr_w != new_w || curr_h != new_h {
                  unsafe { rembrandt_kernel_image3_bicubic_scale(
                      resize_src_buf.as_ref(ctx).as_ptr(),
                      curr_w as i32, curr_h as i32, 3,
                      resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                      new_w as i32, new_h as i32,
                      ctx.stream.ptr,
                  ) };
                }

                let mut dst_buf_h = Vec::with_capacity(new_w * new_h * 3);
                for _ in 0 .. 3 * new_w * new_h {
                  dst_buf_h.push(0.0);
                }

                resize_dst_buf.as_ref_range(0, new_w * new_h * 3, ctx)
                  .sync_store(&mut dst_buf_h);

                let mut resize_im = Image::new(new_w, new_h, 3, repeat(0).take(new_w * new_h * 3).collect());
                for y in 0 .. new_h {
                  for x in 0 .. new_w {
                    for c in 0 .. 3 {
                      resize_im.data[c + x * 3 + y * 3 * new_w]
                          = (dst_buf_h[x + y * new_w + c * new_w * new_h] * 255.0).max(0.0).min(255.0).round() as u8;
                    }
                  }
                }

                {
                  let buf = match resize_im.write_png() {
                    Err(e) => panic!("stb failed to write png: {} {:?}", idx, e),
                    Ok(buf) => buf,
                  };
                  let resize_path = PathBuf::from(&format!("tmp/preproc_debug/{}/valid_{}_resize.png", top_10k, idx));
                  let mut resize_file = File::create(&resize_path).unwrap();
                  resize_file.write_all(&buf).unwrap();
                }

                /*match im.resize(&mut resize_im) {
                  Err(e) => {
                    println!("WARNING: stb failed to resize image: {} {:?}", k, e);
                    continue;
                  }
                  Ok(_) => {}
                }*/

                im = resize_im;
              }

              let mut trans_data = Vec::with_capacity(3 * im.width * im.height);
              for c in 0 .. 3 {
                for y in 0 .. im.height {
                  for x in 0 .. im.width {
                    trans_data.push(im.data[c + x * 3 + y * 3 * im.width]);
                  }
                }
              }
              assert_eq!(im.depth * im.width * im.height, trans_data.len());
              let im_arr = Array3d::<u8>::with_data(trans_data, (im.width, im.height, 3));
              im_arr
            }
          };

          let mut encoded_arr = vec![];
          match image_arr.serialize(&mut encoded_arr) {
            Err(e) => panic!("array serialization error: {:?}", e),
            Ok(_) => {}
          }
          self.writer_tx.send(
              WriterMsg::ImageFile(idx, rawcat, encoded_arr),
          ).unwrap();
        }
      }
    }
    self.writer_tx.send(
        WriterMsg::Quit,
    ).unwrap();
  }
}

struct ValidStbEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl ValidStbEncoderWorker {
  pub fn run(&mut self, dev_idx: usize) {
    let context = DeviceContext::new(dev_idx);
    let ctx = &context.as_ref();
    // XXX(20160505): this takes about 9 GB; yes, some images are large.
    let mut resize_src_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);
    let mut resize_dst_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);

    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          let top_10k = if idx < 10000 {
            0
          } else if idx < 20000 {
            1
          } else if idx < 30000 {
            2
          } else if idx < 40000 {
            3
          } else if idx < 50000 {
            4
          } else {
            unreachable!();
          };
          let image_arr = match load_from_memory(&buf) {
            LoadResult::Error(e) => {
              println!("WARNING: stb failed to load image: {} {:?}", idx, e);
              let im_arr = Array3d::<u8>::zeros((256, 256, 3));
              im_arr
            }
            LoadResult::ImageU8(mut im) => {
              //assert_eq!(3, im.depth);
              if im.depth != 3 && im.depth != 1 {
                panic!("WARNING: stb loaded an unsupported depth: {} {}", idx, im.depth);
              }
              assert_eq!(im.depth * im.width * im.height, im.data.len());

              if im.depth == 1 {
                let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                assert_eq!(im.width * im.height, im.data.len());
                for i in 0 .. im.data.len() {
                  rgb_data.push(im.data[i]);
                  rgb_data.push(im.data[i]);
                  rgb_data.push(im.data[i]);
                }
                assert_eq!(3 * im.width * im.height, rgb_data.len());
                im = Image::new(im.width, im.height, 3, rgb_data);
              }
              assert_eq!(3, im.depth);

              {
                let buf = match im.write_png() {
                  Err(e) => panic!("stb failed to write png: {} {:?}", idx, e),
                  Ok(buf) => buf,
                };
                let orig_path = PathBuf::from(&format!("tmp/preproc_debug/{}/valid_{}_orig.png", top_10k, idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();
              }

              let orig_w = im.width;
              let orig_h = im.height;
              let min_dim = min(orig_w, orig_h);
              if min_dim != 256 {
                let (new_w, new_h) = if min_dim == orig_w {
                  (256, (256 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
                } else {
                  ((256 as f32 / orig_h as f32 * orig_w as f32).round() as usize, 256)
                };

                let mut src_buf_h = Vec::with_capacity(3 * im.width * im.height);
                for c in 0 .. 3 {
                  for y in 0 .. im.height {
                    for x in 0 .. im.width {
                      src_buf_h.push(im.data[c + x * 3 + y * 3 * im.width] as f32 / 255.0);
                    }
                  }
                }

                resize_src_buf.as_ref_mut_range(0, orig_w * orig_h * 3, ctx)
                  .sync_load(&src_buf_h);

                let mut curr_w = orig_w;
                let mut curr_h = orig_h;
                while (curr_w+1)/2 >= new_w && (curr_h+1)/2 >= new_h {
                  unsafe { rembrandt_kernel_image3_bilinear_scale(
                      resize_src_buf.as_ref(ctx).as_ptr(),
                      curr_w as i32, curr_h as i32, 3,
                      resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                      ((curr_w+1)/2) as i32, ((curr_h+1)/2) as i32,
                      ctx.stream.ptr,
                  ) };
                  resize_dst_buf.as_ref(ctx).send(&mut resize_src_buf.as_ref_mut(ctx));
                  curr_w = (curr_w+1)/2;
                  curr_h = (curr_h+1)/2;
                }
                unsafe { rembrandt_kernel_image3_bicubic_scale(
                    resize_src_buf.as_ref(ctx).as_ptr(),
                    curr_w as i32, curr_h as i32, 3,
                    resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                    new_w as i32, new_h as i32,
                    ctx.stream.ptr,
                ) };

                let mut dst_buf_h = Vec::with_capacity(new_w * new_h * 3);
                for _ in 0 .. 3 * new_w * new_h {
                  dst_buf_h.push(0.0);
                }

                resize_dst_buf.as_ref_range(0, new_w * new_h * 3, ctx)
                  .sync_store(&mut dst_buf_h);

                let mut resize_im = Image::new(new_w, new_h, 3, repeat(0).take(new_w * new_h * 3).collect());
                for y in 0 .. new_h {
                  for x in 0 .. new_w {
                    for c in 0 .. 3 {
                      resize_im.data[c + x * 3 + y * 3 * new_w]
                          = (dst_buf_h[x + y * new_w + c * new_w * new_h] * 255.0).max(0.0).min(255.0).round() as u8;
                    }
                  }
                }

                {
                  let buf = match resize_im.write_png() {
                    Err(e) => panic!("stb failed to write png: {} {:?}", idx, e),
                    Ok(buf) => buf,
                  };
                  let resize_path = PathBuf::from(&format!("tmp/preproc_debug/{}/valid_{}_resize.png", top_10k, idx));
                  let mut resize_file = File::create(&resize_path).unwrap();
                  resize_file.write_all(&buf).unwrap();
                }

                /*match im.resize(&mut resize_im) {
                  Err(e) => {
                    println!("WARNING: stb failed to resize image: {} {:?}", k, e);
                    continue;
                  }
                  Ok(_) => {}
                }*/

                im = resize_im;
              }

              let mut trans_data = Vec::with_capacity(3 * im.width * im.height);
              for c in 0 .. 3 {
                for y in 0 .. im.height {
                  for x in 0 .. im.width {
                    trans_data.push(im.data[c + x * 3 + y * 3 * im.width]);
                  }
                }
              }
              assert_eq!(im.depth * im.width * im.height, trans_data.len());
              let im_arr = Array3d::<u8>::with_data(trans_data, (im.width, im.height, 3));
              im_arr
            }
            LoadResult::ImageF32(im) => {
              panic!("WARNING: stb loaded a f32 image: {}", idx);
            }
          };

          let mut encoded_arr = vec![];
          match image_arr.serialize(&mut encoded_arr) {
            Err(e) => panic!("array serialization error: {:?}", e),
            Ok(_) => {}
          }
          self.writer_tx.send(
              WriterMsg::ImageFile(idx, rawcat, encoded_arr),
          ).unwrap();
        }
      }
    }
    self.writer_tx.send(
        WriterMsg::Quit,
    ).unwrap();
  }
}

struct TurboEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl TurboEncoderWorker {
  pub fn run(&mut self, dev_idx: usize) {
    let context = DeviceContext::new(dev_idx);
    let ctx = &context.as_ref();
    // XXX(20160505): this takes about 9 GB; yes, some images are large.
    let mut resize_src_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);
    let mut resize_dst_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);

    let mut decoder = TurbojpegDecoder::create().unwrap();
    let mut encoder = TurbojpegEncoder::create().unwrap();

    let mut max_w = 0;
    let mut max_h = 0;
    let mut max_ratio = 1.0;
    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          {
            let mut image = match decoder.decode_rgb8(&buf) {
              Err(e) => {
                println!("WARNING: turbo decoder failed: {}", idx);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }

              Ok((header, data)) => {
                let mut im = Image::new(header.width, header.height, 3, data);

                if im.depth != 3 && im.depth != 1 {
                  println!("WARNING: stb loaded an unsupported depth: {} {}", idx, im.depth);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                assert_eq!(im.depth * im.width * im.height, im.data.len());

                if im.depth == 1 {
                  let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                  assert_eq!(im.width * im.height, im.data.len());
                  for i in 0 .. im.data.len() {
                    rgb_data.push(im.data[i]);
                    rgb_data.push(im.data[i]);
                    rgb_data.push(im.data[i]);
                  }
                  assert_eq!(3 * im.width * im.height, rgb_data.len());
                  im = Image::new(im.width, im.height, 3, rgb_data);
                }
                assert_eq!(3, im.depth);

                im
              }
            };

            let orig_w = image.width;
            let orig_h = image.height;
            //let orig_pitch = image.pitch();
            if orig_w as f32 / orig_h as f32 > max_ratio {
              max_ratio = orig_w as f32 / orig_h as f32;
              println!("DEBUG: new max ratio (wide): {} {:.3}", idx, max_ratio);
              {
                let div_10k = idx / 10000;
                let dump_dir = PathBuf::from(&format!("tmp/preproc_train_debug/{}", div_10k));
                create_dir_all(&dump_dir).ok();

                let mut orig_path = dump_dir.clone();
                orig_path.push(&format!("train_{}_orig.jpg", idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();
              }
            } else if orig_h as f32 / orig_w as f32 > max_ratio {
              max_ratio = orig_h as f32 / orig_w as f32;
              println!("DEBUG: new max ratio (high): {} {:.3}", idx, max_ratio);
              {
                let div_10k = idx / 10000;
                let dump_dir = PathBuf::from(&format!("tmp/preproc_train_debug/{}", div_10k));
                create_dir_all(&dump_dir).ok();

                let mut orig_path = dump_dir.clone();
                orig_path.push(&format!("train_{}_orig.jpg", idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();
              }
            }
            if orig_w > max_w {
              max_w = orig_w;
              println!("DEBUG: new max width: {} {}", idx, max_w);
            }
            if orig_h > max_h {
              max_h = orig_h;
              println!("DEBUG: new max width: {} {}", idx, max_h);
            }

            let min_dim = min(orig_w, orig_h);

            let encoded_buf = if min_dim <= 480 {
              buf

            } else {
              let (new_w, new_h) = if min_dim == orig_w {
                (480, (480 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
              } else {
                ((480 as f32 / orig_h as f32 * orig_w as f32).round() as usize, 480)
              };

              let mut src_buf_h = Vec::with_capacity(3 * image.width * image.height);
              for c in 0 .. 3 {
                for y in 0 .. image.height {
                  for x in 0 .. image.width {
                    src_buf_h.push(image.data[c + x * 3 + y * 3 * image.width] as f32 / 255.0);
                  }
                }
              }

              resize_src_buf.as_ref_mut_range(0, orig_w * orig_h * 3, ctx)
                .sync_load(&src_buf_h);

              /*unsafe { rembrandt_kernel_image3_bicubic_scale(
                  resize_src_buf.as_ref(ctx).as_ptr(),
                  orig_w as i32, orig_h as i32, 3,
                  resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                  new_w as i32, new_h as i32,
                  ctx.stream.ptr,
              ) };*/
              let mut curr_w = orig_w;
              let mut curr_h = orig_h;
              while (curr_w+1)/2 >= new_w && (curr_h+1)/2 >= new_h {
                unsafe { rembrandt_kernel_image3_bilinear_scale(
                    resize_src_buf.as_ref(ctx).as_ptr(),
                    curr_w as i32, curr_h as i32, 3,
                    resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                    ((curr_w+1)/2) as i32, ((curr_h+1)/2) as i32,
                    ctx.stream.ptr,
                ) };
                resize_dst_buf.as_ref(ctx).send(&mut resize_src_buf.as_ref_mut(ctx));
                curr_w = (curr_w+1)/2;
                curr_h = (curr_h+1)/2;
              }
              assert!(curr_w >= new_w);
              assert!(curr_h >= new_h);
              if curr_w > new_w || curr_h > new_h {
                unsafe { rembrandt_kernel_image3_bicubic_scale(
                    resize_src_buf.as_ref(ctx).as_ptr(),
                    curr_w as i32, curr_h as i32, 3,
                    resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                    new_w as i32, new_h as i32,
                    ctx.stream.ptr,
                ) };
              } else {
                assert_eq!(curr_w, new_w);
                assert_eq!(curr_h, new_h);
              }

              let mut dst_buf_h = Vec::with_capacity(new_w * new_h * 3);
              for _ in 0 .. 3 * new_w * new_h {
                dst_buf_h.push(0.0);
              }

              resize_dst_buf.as_ref_range(0, new_w * new_h * 3, ctx)
                .sync_store(&mut dst_buf_h);

              let mut resize_image = Image::new(new_w, new_h, 3, repeat(0).take(new_w * new_h * 3).collect());
              for y in 0 .. new_h {
                for x in 0 .. new_w {
                  for c in 0 .. 3 {
                    resize_image.data[c + x * 3 + y * 3 * new_w]
                        = (dst_buf_h[x + y * new_w + c * new_w * new_h] * 255.0).max(0.0).min(255.0).round() as u8;
                  }
                }
              }

              let mut encoded_buf = match encoder.encode_rgb8(&resize_image.data, resize_image.width, resize_image.height) {
                Err(e) => {
                  println!("WARNING: turbo failed encode: {}", idx);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                Ok(buf) => buf,
              };

              /*// FIXME(20160505): encode resized image as PNG.
              let mut resize_imagebuf = match ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(new_w as u32, new_h as u32, resize_image.data) {
                None => {
                  println!("WARNING: failed to make piston imagebuffer from raw bytes: {}", idx);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                Some(imagebuf) => imagebuf,
              };

              let mut encoded_buf = vec![];
              let dynimage = DynamicImage::ImageRgb8(resize_imagebuf);
              match dynimage.save(&mut encoded_buf, ImageFormat::JPEG) {
                Err(e) => {
                  println!("WARNING: failed to encode old image as jpg: {} {:?}", idx, e);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                Ok(_) => {}
              }*/

              {
                let div_10k = idx / 10000;
                let dump_dir = PathBuf::from(&format!("tmp/preproc_train_debug/{}", div_10k));
                create_dir_all(&dump_dir).ok();
                //let resize_path = PathBuf::from(&format!("tmp/preproc_train_debug/{}/valid_{}_resize.png", div_10k, idx));

                let mut orig_path = dump_dir.clone();
                orig_path.push(&format!("train_{}_orig.jpg", idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();

                let mut resize_path = dump_dir.clone();
                resize_path.push(&format!("train_{}_resize.jpg", idx));
                let mut resize_file = File::create(&resize_path).unwrap();
                resize_file.write_all(&encoded_buf).unwrap();
              }

              /*{
                let mut test_reader = Cursor::new(&encoded_buf);
                match load(test_reader, ImageFormat::JPEG) {
                  Err(e) => {
                    println!("WARNING: failed to decode the encoded image as jpg: {} {:?}", idx, e);
                    self.writer_tx.send(
                        WriterMsg::Skip(idx),
                    ).unwrap();
                    continue;
                  }
                  Ok(_) => {}
                }
              }

              {
                let mut test_image = match load_from_memory(&encoded_buf) {
                  LoadResult::ImageU8(im) => im,
                  _ => {
                    println!("WARNING: failed to decode the encoded image as jpg (stb): {}", idx);
                    self.writer_tx.send(
                        WriterMsg::Skip(idx),
                    ).unwrap();
                    continue;
                  }
                };
                assert_eq!(new_w, test_image.width);
                assert_eq!(new_h, test_image.height);
                assert_eq!(3, test_image.depth);
              }*/

              {
                match decoder.decode_rgb8(&encoded_buf) {
                  Err(e) => {
                    println!("WARNING: turbo decoder failed to decode encoded jpeg: {}", idx);
                    self.writer_tx.send(
                        WriterMsg::Skip(idx),
                    ).unwrap();
                    continue;
                  }
                  Ok((header, data)) => {
                    assert_eq!(new_w, header.width);
                    assert_eq!(new_h, header.height);
                    assert_eq!(new_w * new_h * 3, data.len());
                  }
                }
              }

              encoded_buf
            };

            self.writer_tx.send(
                WriterMsg::ImageFile(idx, rawcat, encoded_buf),
            ).unwrap();
          }
        }
      }
    }
    println!("DEBUG: encoder: max_w: {} max_h: {} max_ratio: {:.3}", max_w, max_h, max_ratio);
    self.writer_tx.send(
        WriterMsg::Quit,
    ).unwrap();
  }
}

struct StbEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl StbEncoderWorker {
  pub fn run(&mut self, dev_idx: usize) {
    let context = DeviceContext::new(dev_idx);
    let ctx = &context.as_ref();
    // XXX(20160505): this takes about 9 GB; yes, some images are large.
    let mut resize_src_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);
    let mut resize_dst_buf = DeviceBuffer::<f32>::zeros(384 * 1024 * 1024 * 3, ctx);

    let mut max_w = 0;
    let mut max_h = 0;
    let mut max_ratio = 1.0;
    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          {
            let mut image = match load_from_memory(&buf) {
              LoadResult::Error(e) => {
                println!("WARNING: stb failed to load jpg: {} {:?}", idx, e);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }
              LoadResult::ImageU8(mut im) => {
                if im.depth != 3 && im.depth != 1 {
                  println!("WARNING: stb loaded an unsupported depth: {} {}", idx, im.depth);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                assert_eq!(im.depth * im.width * im.height, im.data.len());

                if im.depth == 1 {
                  let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                  assert_eq!(im.width * im.height, im.data.len());
                  for i in 0 .. im.data.len() {
                    rgb_data.push(im.data[i]);
                    rgb_data.push(im.data[i]);
                    rgb_data.push(im.data[i]);
                  }
                  assert_eq!(3 * im.width * im.height, rgb_data.len());
                  im = Image::new(im.width, im.height, 3, rgb_data);
                }
                assert_eq!(3, im.depth);

                im
              }
              LoadResult::ImageF32(im) => {
                println!("WARNING: stb loaded a f32 image: {}", idx);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }
            };

            let orig_w = image.width;
            let orig_h = image.height;
            //let orig_pitch = image.pitch();
            if orig_w as f32 / orig_h as f32 > max_ratio {
              max_ratio = orig_w as f32 / orig_h as f32;
              println!("DEBUG: new max ratio (wide): {} {:.3}", idx, max_ratio);
              {
                let div_10k = idx / 10000;
                let dump_dir = PathBuf::from(&format!("tmp/preproc_train_debug/{}", div_10k));
                create_dir_all(&dump_dir).ok();

                let mut orig_path = dump_dir.clone();
                orig_path.push(&format!("train_{}_orig.jpg", idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();
              }
            } else if orig_h as f32 / orig_w as f32 > max_ratio {
              max_ratio = orig_h as f32 / orig_w as f32;
              println!("DEBUG: new max ratio (high): {} {:.3}", idx, max_ratio);
              {
                let div_10k = idx / 10000;
                let dump_dir = PathBuf::from(&format!("tmp/preproc_train_debug/{}", div_10k));
                create_dir_all(&dump_dir).ok();

                let mut orig_path = dump_dir.clone();
                orig_path.push(&format!("train_{}_orig.jpg", idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();
              }
            }
            if orig_w > max_w {
              max_w = orig_w;
              println!("DEBUG: new max width: {} {}", idx, max_w);
            }
            if orig_h > max_h {
              max_h = orig_h;
              println!("DEBUG: new max width: {} {}", idx, max_h);
            }

            let min_dim = min(orig_w, orig_h);

            let encoded_buf = if min_dim <= 480 {
              buf

            } else {
              let (new_w, new_h) = if min_dim == orig_w {
                (480, (480 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
              } else {
                ((480 as f32 / orig_h as f32 * orig_w as f32).round() as usize, 480)
              };

              let mut src_buf_h = Vec::with_capacity(3 * image.width * image.height);
              for c in 0 .. 3 {
                for y in 0 .. image.height {
                  for x in 0 .. image.width {
                    src_buf_h.push(image.data[c + x * 3 + y * 3 * image.width] as f32 / 255.0);
                  }
                }
              }

              resize_src_buf.as_ref_mut_range(0, orig_w * orig_h * 3, ctx)
                .sync_load(&src_buf_h);

              /*unsafe { rembrandt_kernel_image3_bicubic_scale(
                  resize_src_buf.as_ref(ctx).as_ptr(),
                  orig_w as i32, orig_h as i32, 3,
                  resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                  new_w as i32, new_h as i32,
                  ctx.stream.ptr,
              ) };*/
              let mut curr_w = orig_w;
              let mut curr_h = orig_h;
              while (curr_w+1)/2 >= new_w && (curr_h+1)/2 >= new_h {
                unsafe { rembrandt_kernel_image3_bilinear_scale(
                    resize_src_buf.as_ref(ctx).as_ptr(),
                    curr_w as i32, curr_h as i32, 3,
                    resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                    ((curr_w+1)/2) as i32, ((curr_h+1)/2) as i32,
                    ctx.stream.ptr,
                ) };
                resize_dst_buf.as_ref(ctx).send(&mut resize_src_buf.as_ref_mut(ctx));
                curr_w = (curr_w+1)/2;
                curr_h = (curr_h+1)/2;
              }
              assert!(curr_w >= new_w);
              assert!(curr_h >= new_h);
              if curr_w > new_w || curr_h > new_h {
                unsafe { rembrandt_kernel_image3_bicubic_scale(
                    resize_src_buf.as_ref(ctx).as_ptr(),
                    curr_w as i32, curr_h as i32, 3,
                    resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                    new_w as i32, new_h as i32,
                    ctx.stream.ptr,
                ) };
              }

              let mut dst_buf_h = Vec::with_capacity(new_w * new_h * 3);
              for _ in 0 .. 3 * new_w * new_h {
                dst_buf_h.push(0.0);
              }

              resize_dst_buf.as_ref_range(0, new_w * new_h * 3, ctx)
                .sync_store(&mut dst_buf_h);

              let mut resize_image = Image::new(new_w, new_h, 3, repeat(0).take(new_w * new_h * 3).collect());
              for y in 0 .. new_h {
                for x in 0 .. new_w {
                  for c in 0 .. 3 {
                    resize_image.data[c + x * 3 + y * 3 * new_w]
                        = (dst_buf_h[x + y * new_w + c * new_w * new_h] * 255.0).max(0.0).min(255.0).round() as u8;
                  }
                }
              }

              // FIXME(20160505): encode resized image as PNG.
              let mut resize_imagebuf = match ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(new_w as u32, new_h as u32, resize_image.data) {
                None => {
                  println!("WARNING: failed to make piston imagebuffer from raw bytes: {}", idx);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                Some(imagebuf) => imagebuf,
              };

              let mut encoded_buf = vec![];
              let dynimage = DynamicImage::ImageRgb8(resize_imagebuf);
              match dynimage.save(&mut encoded_buf, ImageFormat::JPEG) {
                Err(e) => {
                  println!("WARNING: failed to encode old image as jpg: {} {:?}", idx, e);
                  self.writer_tx.send(
                      WriterMsg::Skip(idx),
                  ).unwrap();
                  continue;
                }
                Ok(_) => {}
              }

              {
                let div_10k = idx / 10000;
                let dump_dir = PathBuf::from(&format!("tmp/preproc_train_debug/{}", div_10k));
                create_dir_all(&dump_dir).ok();
                //let resize_path = PathBuf::from(&format!("tmp/preproc_train_debug/{}/valid_{}_resize.png", div_10k, idx));

                let mut orig_path = dump_dir.clone();
                orig_path.push(&format!("train_{}_orig.jpg", idx));
                let mut orig_file = File::create(&orig_path).unwrap();
                orig_file.write_all(&buf).unwrap();

                let mut resize_path = dump_dir.clone();
                resize_path.push(&format!("train_{}_resize.jpg", idx));
                let mut resize_file = File::create(&resize_path).unwrap();
                resize_file.write_all(&encoded_buf).unwrap();
              }

              {
                let mut test_reader = Cursor::new(&encoded_buf);
                match load(test_reader, ImageFormat::JPEG) {
                  Err(e) => {
                    println!("WARNING: failed to decode the encoded image as jpg: {} {:?}", idx, e);
                    self.writer_tx.send(
                        WriterMsg::Skip(idx),
                    ).unwrap();
                    continue;
                  }
                  Ok(_) => {}
                }
              }

              {
                let mut test_image = match load_from_memory(&encoded_buf) {
                  LoadResult::ImageU8(im) => im,
                  _ => {
                    println!("WARNING: failed to decode the encoded image as jpg (stb): {}", idx);
                    self.writer_tx.send(
                        WriterMsg::Skip(idx),
                    ).unwrap();
                    continue;
                  }
                };
                assert_eq!(new_w, test_image.width);
                assert_eq!(new_h, test_image.height);
                assert_eq!(3, test_image.depth);
              }

              encoded_buf
            };

            self.writer_tx.send(
                WriterMsg::ImageFile(idx, rawcat, encoded_buf),
            ).unwrap();
          }
        }
      }
    }
    println!("DEBUG: encoder: max_w: {} max_h: {} max_ratio: {:.3}", max_w, max_h, max_ratio);
    self.writer_tx.send(
        WriterMsg::Quit,
    ).unwrap();
  }
}

struct SdlEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl SdlEncoderWorker {
  pub fn run(&mut self) {
    let template_surf = match Surface::new(480, 480, PixelFormatEnum::RGB888) {
      Err(e) => panic!("failed to make template surface"),
      Ok(surf) => surf,
    };
    let mut max_w = 0;
    let mut max_h = 0;
    let mut max_ratio = 1.0;
    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          /*self.writer_tx.send(
              WriterMsg::Quit,
          ).unwrap();*/
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          /*if idx < 100 {
            let mut debug_file = File::create(&PathBuf::from(&format!("tmp/imagenet_{}.jpg", idx))).unwrap();
            debug_file.write_all(&buf).unwrap();
          }*/
          {
            let rw = match RWops::from_bytes(&buf) {
              Err(e) => {
                println!("WARNING: sdl rwops error: {} {:?}", idx, e);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }
              Ok(rw) => rw,
            };
            let image_surf = match rw.load_jpg() {
              Err(e) => {
                println!("WARNING: sdl decode error: {} {:?}", idx, e);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }
              Ok(surf) => surf,
            };
            //let (orig_w, orig_h) = image.get_size();
            let orig_w = image_surf.width();
            let orig_h = image_surf.height();
            let orig_pitch = image_surf.pitch();
            if orig_w as f32 / orig_h as f32 > max_ratio {
              max_ratio = orig_w as f32 / orig_h as f32;
              println!("DEBUG: new max ratio: {} {:.3}", idx, max_ratio);
            } else if orig_h as f32 / orig_w as f32 > max_ratio {
              max_ratio = orig_h as f32 / orig_w as f32;
              println!("DEBUG: new max ratio: {} {:.3}", idx, max_ratio);
            }
            if orig_w > max_w {
              max_w = orig_w;
              println!("DEBUG: new max width: {} {}", idx, max_w);
            }
            if orig_h > max_h {
              max_h = orig_h;
              println!("DEBUG: new max width: {} {}", idx, max_h);
            }

            let mut target_surf = match Surface::new(orig_w, orig_h, PixelFormatEnum::RGB888) {
              Err(e) => {
                println!("WARNING: failed to create target surface: {} {:?}", idx, e);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }
              Ok(surf) => surf,
            };
            match image_surf.blit(None, &mut target_surf, None) {
              Err(e) => {
                println!("WARNING: failed to blit: {} {:?}", idx, e);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
              }
              Ok(_) => {}
            }

            if target_surf.must_lock() {
              target_surf.with_lock(|pixels| {
              });
            } else {
              target_surf.without_lock().map(|pixels| {
              }).unwrap();
            }

            let min_dim = min(orig_w, orig_h);

            // TODO
          }

          self.writer_tx.send(
              WriterMsg::ImageFile(idx, rawcat, buf),
          ).unwrap();
        }
      }
    }
    println!("DEBUG: encoder: max_w: {} max_h: {} max_ratio: {:.3}", max_w, max_h, max_ratio);
    self.writer_tx.send(
        WriterMsg::Quit,
    ).unwrap();
  }
}

struct EpegEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl EpegEncoderWorker {
  pub fn run(&mut self) {
    //let _vips = Vips::new();
    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          self.writer_tx.send(
              WriterMsg::Quit,
          ).unwrap();
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          /*if idx < 100 {
            let mut debug_file = File::create(&PathBuf::from(&format!("tmp/imagenet_{}.jpg", idx))).unwrap();
            debug_file.write_all(&buf).unwrap();
          }*/
          let mut image = match EpegImage::open_memory(buf.clone()) {
            Err(e) => {
              println!("WARNING: epeg decode error: {}", idx);
              self.writer_tx.send(
                  WriterMsg::Skip(idx),
              ).unwrap();
              continue;
            }
            Ok(im) => im,
          };
          let (orig_w, orig_h) = image.get_size();
          let min_dim = min(orig_w, orig_h);
          let (new_w, new_h) = if min_dim == orig_w {
            (480, (480 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
          } else {
            ((480 as f32 / orig_w as f32 * orig_w as f32).round() as usize, 480)
          };
          image.set_decode_size(new_w, new_h);
          let mut out_buf = Vec::with_capacity(3 * new_w * new_h);
          unsafe { out_buf.set_len(3 * new_w * new_h) };
          match image.scale_to_memory(&mut out_buf) {
            Err(e) => {
              println!("WARNING: epeg scale error: {:?}", e);
              self.writer_tx.send(
                  WriterMsg::Skip(idx),
              ).unwrap();
              continue;
            }
            Ok(_) => {}
          }
          self.writer_tx.send(
              WriterMsg::ImageFile(idx, rawcat, buf),
          ).unwrap();
        }
      }
    }
  }
}

/*struct VipsEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl VipsEncoderWorker {
  pub fn run(&mut self) {
    loop {
      match self.encoder_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(EncoderMsg::Quit) => {
          self.writer_tx.send(
              WriterMsg::Quit,
          ).unwrap();
          break;
        }

        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          {
            let mut debug_file = File::create(&PathBuf::from(&format!("tmp/imagenet_{}.jpg", idx))).unwrap();
            debug_file.write_all(&buf).unwrap();
          }
          //let buf = buf.to_vec();
          let image = match VipsImage::decode(buf) {
            Err(e) => {
              println!("WARNING: vips decode error: {}", idx);
              continue;
            }
            Ok(im) => im,
          };
        }
      }
    }
  }
}*/

struct PistonEncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl PistonEncoderWorker {
  pub fn run(&mut self) {
    loop {
      match self.encoder_rx.recv() {
        //Err(_) | Ok(EncoderMsg::Quit) => {
        Err(_) => {
          break;
        }
        Ok(EncoderMsg::Quit) => {
          self.writer_tx.send(
              WriterMsg::Quit,
          ).unwrap();
          break;
        }
        Ok(EncoderMsg::RawImageFile(idx, rawcat, buf)) => {
          let mut reader = Cursor::new(&buf);
          let mut image = None;
          match load(reader, ImageFormat::JPEG) {
            Err(_) => {}
            Ok(im) => image = Some(im),
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::PNG) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::GIF) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::TIFF) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::BMP) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            println!("WARNING: decoding failed: {}", idx);
            {
              let mut debug_file = File::create(&PathBuf::from(&format!("imagenet_{}.jpg", idx))).unwrap();
              debug_file.write_all(&buf).unwrap();
            }
            self.writer_tx.send(
                WriterMsg::Skip(idx),
            ).unwrap();
            continue;
          }
          continue; // FIXME(20160503): for debugging.

          let image = image.unwrap();
          let image_dims = image.dimensions();
          let min_side = min(image_dims.0, image_dims.1);
          //println!("DEBUG: min side: {} dims: {:?} path: {:?}", min_side, image_dims, im_path);

          let mut is_new = false;
          let mut encoded_image = vec![];
          if let Some(resize_smaller_dim) = self.config.resize_smaller_dim {
            if min_side <= resize_smaller_dim {
              /*let image_buf = match image.as_rgb8() {
                Some(buf) => buf.clone(),
                None => panic!("failed to interpret image as rgb8"),
              };*/
              let image_buf = image.to_rgb();
              let image = DynamicImage::ImageRgb8(image_buf);
              match image.save(&mut encoded_image, ImageFormat::PNG) {
                Err(e) => panic!("failed to encode old image as png: {:?}", e),
                Ok(_) => {}
              }
            } else {
              is_new = true;

              let (old_width, old_height) = image_dims;
              let (new_width, new_height) = if old_width < old_height {
                (min_side, (min_side as f32 / old_width as f32 * old_height as f32).round() as u32)
              } else if old_width > old_height {
                ((min_side as f32 / old_height as f32 * old_width as f32).round() as u32, min_side)
              } else {
                (min_side, min_side)
              };

              /*let old_image_buf = match image.as_rgb8() {
                Some(buf) => buf.clone(),
                None => panic!("failed to interpret image as rgb8"),
              };*/
              let old_image_buf = image.to_rgb();
              let new_image_buf = resize(&old_image_buf, new_width as u32, new_height as u32, FilterType::Lanczos3);

              let new_image = DynamicImage::ImageRgb8(new_image_buf);
              match new_image.save(&mut encoded_image, ImageFormat::PNG) {
                Err(e) => panic!("failed to encode new image as png: {:?}", e),
                Ok(_) => {}
              }
            }
          } else {
            let image_buf = image.to_rgb();
            let image = DynamicImage::ImageRgb8(image_buf);
            match image.save(&mut encoded_image, ImageFormat::PNG) {
              Err(e) => panic!("failed to encode old image as png: {:?}", e),
              Ok(_) => {}
            }
          }

          {
            let mut test_reader = Cursor::new(&encoded_image);
            match load(test_reader, ImageFormat::PNG) {
              Err(e) => {
                println!("WARNING: failed to decode the (new? {:?}) encoded image as png: {} {:?}", is_new, idx, e);
                self.writer_tx.send(
                    WriterMsg::Skip(idx),
                ).unwrap();
                continue;
              }
              Ok(_) => {}
            }
          }

          self.writer_tx.send(
              WriterMsg::ImageFile(idx, rawcat, encoded_image),
          ).unwrap();
        }
      }
    }
  }
}

struct WriterWorker {
  config:       IlsvrcConfig,
  writer_rx:    Receiver<WriterMsg>,
  counter:      usize,
}

impl WriterWorker {
  pub fn run(&mut self) {
    let mut quit_count = 0;
    //let mut adj_idx = 0;
    let mut skip_set: HashSet<usize> = HashSet::new();
    let mut cache: BTreeMap<usize, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
    let mut data_db = VarrayDb::create(&self.config.train_data_path).unwrap();
    let mut labels_db = VarrayDb::create(&self.config.train_labels_path).unwrap();
    loop {
      match self.writer_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(WriterMsg::Quit) => {
          quit_count += 1;
          if quit_count >= NUM_ENCODERS {
            assert!(cache.is_empty());
            println!("DEBUG: total number skipped: {}", skip_set.len());
            break;
          }
        }

        Ok(WriterMsg::Skip(idx)) => {
          skip_set.insert(idx);
          if idx == self.counter {
            //adj_idx += 1;
            self.counter += 1;
          }
          loop {
            if cache.contains_key(&self.counter) {
              let (buf, label_buf) = cache.remove(&self.counter).unwrap();
              data_db.append(&buf);
              labels_db.append(&label_buf);
              self.counter += 1;
            } else if skip_set.contains(&self.counter) {
              //adj_idx += 1;
              self.counter += 1;
            } else {
              break;
            }
          }
        }

        Ok(WriterMsg::ImageFile(idx, rawcat, buf)) => {
          //println!("DEBUG: images writen: {}", self.counter);
          let label_cat = rawcat - 1;
          assert!(label_cat >= 0 && label_cat < 1000);
          let mut label_buf = vec![];
          label_buf.write_i32::<LittleEndian>(label_cat).unwrap();
          assert_eq!(4, label_buf.len());

          if idx == self.counter {
            assert!(!cache.contains_key(&self.counter));
            data_db.append(&buf);
            labels_db.append(&label_buf);
            self.counter += 1;
          } else {
            cache.insert(idx, (buf, label_buf));
          }
          loop {
            if cache.contains_key(&self.counter) {
              let (buf, label_buf) = cache.remove(&self.counter).unwrap();
              data_db.append(&buf);
              labels_db.append(&label_buf);
              self.counter += 1;
            } else if skip_set.contains(&self.counter) {
              //adj_idx += 1;
              self.counter += 1;
            } else {
              break;
            }
          }

          /*if self.counter % 100 == 0 {
            println!("DEBUG: images writen: {}", self.counter);
          }*/
        }
      }
    }
  }
}

struct ValidWriterWorker {
  config:       IlsvrcConfig,
  writer_rx:    Receiver<WriterMsg>,
  counter:      usize,
}

impl ValidWriterWorker {
  pub fn run(&mut self) {
    let mut quit_count = 0;
    //let mut adj_idx = 0;
    let mut skip_set: HashSet<usize> = HashSet::new();
    let mut cache: BTreeMap<usize, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
    //let mut data_db = VarrayDb::create(&self.config.train_data_path).unwrap();
    //let mut labels_db = VarrayDb::create(&self.config.train_labels_path).unwrap();
    let mut data_db = VarrayDb::create(&self.config.valid_data_path).unwrap();
    let mut labels_db = VarrayDb::create(&self.config.valid_labels_path).unwrap();
    loop {
      match self.writer_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(WriterMsg::Quit) => {
          quit_count += 1;
          if quit_count >= NUM_ENCODERS {
            assert!(cache.is_empty());
            println!("DEBUG: total number skipped: {}", skip_set.len());
            break;
          }
        }

        Ok(WriterMsg::Skip(idx)) => {
          skip_set.insert(idx);
          if idx == self.counter {
            //adj_idx += 1;
            self.counter += 1;
          }
          loop {
            if cache.contains_key(&self.counter) {
              let (buf, label_buf) = cache.remove(&self.counter).unwrap();
              data_db.append(&buf);
              labels_db.append(&label_buf);
              self.counter += 1;
            } else if skip_set.contains(&self.counter) {
              //adj_idx += 1;
              self.counter += 1;
            } else {
              break;
            }
          }
        }

        Ok(WriterMsg::ImageFile(idx, rawcat, buf)) => {
          //println!("DEBUG: images writen: {}", self.counter);
          let label_cat = rawcat - 1;
          assert!(label_cat >= 0 && label_cat < 1000);
          let mut label_buf = vec![];
          label_buf.write_i32::<LittleEndian>(label_cat).unwrap();
          assert_eq!(4, label_buf.len());

          if idx == self.counter {
            assert!(!cache.contains_key(&self.counter));
            data_db.append(&buf);
            labels_db.append(&label_buf);
            self.counter += 1;
          } else {
            cache.insert(idx, (buf, label_buf));
          }
          loop {
            if cache.contains_key(&self.counter) {
              let (buf, label_buf) = cache.remove(&self.counter).unwrap();
              data_db.append(&buf);
              labels_db.append(&label_buf);
              self.counter += 1;
            } else if skip_set.contains(&self.counter) {
              //adj_idx += 1;
              self.counter += 1;
            } else {
              break;
            }
          }

          /*if self.counter % 100 == 0 {
            println!("DEBUG: images writen: {}", self.counter);
          }*/
        }
      }
    }
  }
}

impl IlsvrcConfig {
//pub fn preproc_archive(config: &IlsvrcConfig) {
  pub fn preproc_train_data(&self) {
    let mut wnid_to_rawcats = HashMap::with_capacity(1000);
    {
      let mut reader = CsvReader::from_file(&self.wnid_to_rawcats_path).unwrap();
      for record in reader.decode() {
        let record: WnidRawCatRecord = record.unwrap();
        wnid_to_rawcats.insert(record.wnid, record.raw_cat);
      }
    }
    //println!("DEBUG: {:?}", wnid_to_rawcats);

    let archive_file = File::open(&self.train_archive_path).unwrap();
    let archive_map = Mmap::open(&archive_file, Protection::Read).unwrap();

    let mut samples_count = 0;
    {
      //let train_archive_file = File::open(&self.train_archive_path).unwrap();
      //let mut train_archive = Archive::new(train_archive_file);
      let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
      let mut train_archive = Archive::new(reader);
      for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
        let wnid_file = wnid_file.unwrap();
        let wnid_path: PathBuf = wnid_file.header().path().unwrap().into_owned();
        //println!("DEBUG: counting {} {:?}...", wnid_idx, wnid_path);
        let mut wnid_archive = Archive::new(wnid_file);
        for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
          let im_file = im_file.unwrap();
          samples_count += 1;
        }
      }
    }
    println!("DEBUG: samples count: {}", samples_count);

    //let mut shuffle_idxs: Vec<_> = (0 .. samples_count).collect();
    //thread_rng().shuffle(&mut shuffle_idxs);

    let mut encoder_txs = vec![];
    let mut encoder_rxs = vec![];
    for _ in 0 .. NUM_ENCODERS {
      let (encoder_tx, encoder_rx) = sync_channel(128);
      encoder_txs.push(encoder_tx);
      encoder_rxs.push(Some(encoder_rx));
    }
    let (writer_tx, writer_rx) = sync_channel(128);

    let reader_thr = {
      let config = self.clone();
      spawn(move || {
        ReaderWorker{
          config:       config,
          encoder_txs:  encoder_txs,
          wnid_to_rawcats:  wnid_to_rawcats,
        }.run(archive_map);
      })
    };

    //let _vips = Vips::new();
    let encoder_barrier = Arc::new(Barrier::new(NUM_ENCODERS + 1));
    let encoder_pool = ThreadPool::new(NUM_ENCODERS);
    for i in 0 .. NUM_ENCODERS {
      let encoder_barrier = encoder_barrier.clone();
      let config = self.clone();
      let encoder_rx = encoder_rxs[i].take().unwrap();
      let writer_tx = writer_tx.clone();
      encoder_pool.execute(move || {
        //PistonEncoderWorker{
        //VipsEncoderWorker{
        //EpegEncoderWorker{
        //SdlEncoderWorker{
        //StbEncoderWorker{
        TurboEncoderWorker{
          config:     config,
          encoder_rx: encoder_rx,
          writer_tx:  writer_tx,
        }.run(i);
        encoder_barrier.wait();
      });
    }

    let writer_thr = {
      let config = self.clone();
      spawn(move || {
        WriterWorker{
          config:     config,
          writer_rx:  writer_rx,
          counter:    0,
        }.run();
      })
    };

    writer_thr.join();
    encoder_barrier.wait();
    reader_thr.join();
  }

  pub fn preproc_valid_data(&self) {
    let mut valid_rawcats = Vec::with_capacity(50_000);
    {
      //let mut reader = CsvReader::from_file(&self.wnid_to_rawcats_path).unwrap();
      let reader = BufReader::new(File::open(&self.valid_rawcats_path).unwrap());
      for line in reader.lines() {
        let line = line.unwrap();
        let rawcat: i32 = line.parse().unwrap();
        valid_rawcats.push(rawcat);
      }
    }
    assert_eq!(50_000, valid_rawcats.len());

    let archive_file = File::open(&self.valid_archive_path).unwrap();
    let archive_map = Mmap::open(&archive_file, Protection::Read).unwrap();

    let mut samples_count = 0;
    {
      let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
      let mut archive = Archive::new(reader);
      for (im_idx, im_file) in archive.entries().unwrap().enumerate() {
        let im_file = im_file.unwrap();
        samples_count += 1;
      }
    }
    println!("DEBUG: samples count: {}", samples_count);
    assert_eq!(50_000, samples_count);

    let mut encoder_txs = vec![];
    let mut encoder_rxs = vec![];
    for _ in 0 .. NUM_ENCODERS {
      let (encoder_tx, encoder_rx) = sync_channel(128);
      encoder_txs.push(encoder_tx);
      encoder_rxs.push(Some(encoder_rx));
    }
    let (writer_tx, writer_rx) = sync_channel(128);

    let reader_thr = {
      let config = self.clone();
      spawn(move || {
        ValidReaderWorker{
          config:       config,
          encoder_txs:  encoder_txs,
          valid_rawcats:    valid_rawcats,
        }.run(archive_map);
      })
    };

    let encoder_barrier = Arc::new(Barrier::new(NUM_ENCODERS + 1));
    let encoder_pool = ThreadPool::new(NUM_ENCODERS);
    for i in 0 .. NUM_ENCODERS {
      let encoder_barrier = encoder_barrier.clone();
      let config = self.clone();
      let encoder_rx = encoder_rxs[i].take().unwrap();
      let writer_tx = writer_tx.clone();
      encoder_pool.execute(move || {
        //ValidStbEncoderWorker{
        ValidTurboEncoderWorker{
          config:     config,
          encoder_rx: encoder_rx,
          writer_tx:  writer_tx,
        }.run(i);
        encoder_barrier.wait();
      });
    }

    let writer_thr = {
      let config = self.clone();
      spawn(move || {
        ValidWriterWorker{
          config:     config,
          writer_rx:  writer_rx,
          counter:    0,
        }.run();
      })
    };

    writer_thr.join();
    encoder_barrier.wait();
    reader_thr.join();

    /*let mut data_db = VarrayDb::create(&self.valid_data_path).unwrap();
    let mut labels_db = VarrayDb::create(&self.valid_labels_path).unwrap();

    let context = DeviceContext::new(0);
    let ctx = &context.as_ref();
    let mut resize_src_buf = DeviceBuffer::<f32>::zeros(4096 * 4096 * 3, ctx);
    let mut resize_dst_buf = DeviceBuffer::<f32>::zeros(4096 * 4096 * 3, ctx);

    let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
    let mut archive = Archive::new(reader);
    let mut counter = 0;
    let start_time = get_time();
    for (k, im_file) in archive.entries().unwrap().enumerate() {
      let mut im_file = im_file.unwrap();

      let im_path = im_file.header().path().unwrap().into_owned();
      let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
      let im_stem_toks: Vec<_> = im_path_toks[0].splitn(3, "_").collect();
      let idx: usize = im_stem_toks[2].parse().unwrap();

      let rawcat = valid_rawcats[idx-1];
      let label_cat = rawcat - 1;
      assert!(label_cat >= 0 && label_cat < 1000);
      let mut label_buf = vec![];
      label_buf.write_i32::<LittleEndian>(label_cat).unwrap();
      assert_eq!(4, label_buf.len());

      let mut buf: Vec<u8> = vec![];
      match im_file.read_to_end(&mut buf) {
        Err(e) => panic!("failed to read image: {} {:?}", k, e),
        Ok(_) => {}
      }

      {
        let image_arr = match load_from_memory(&buf) {
          LoadResult::Error(e) => {
            println!("WARNING: stb failed to load image: {} {:?}", k, e);
            let im_arr = Array3d::<u8>::zeros((256, 256, 3));
            im_arr
          }
          LoadResult::ImageU8(mut im) => {
            //assert_eq!(3, im.depth);
            if im.depth != 3 && im.depth != 1 {
              println!("WARNING: stb loaded an unsupported depth: {} {}", k, im.depth);
              continue;
            }
            assert_eq!(im.depth * im.width * im.height, im.data.len());

            if im.depth == 1 {
              let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
              assert_eq!(im.width * im.height, im.data.len());
              for i in 0 .. im.data.len() {
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
              }
              assert_eq!(3 * im.width * im.height, rgb_data.len());
              im = Image::new(im.width, im.height, 3, rgb_data);
            }
            assert_eq!(3, im.depth);

            {
              let buf = match im.write_png() {
                Err(e) => panic!("stb failed to write png: {} {:?}", k, e),
                Ok(buf) => buf,
              };
              let orig_path = PathBuf::from(&format!("tmp/preproc_debug/valid_{}_orig.png", k));
              let mut orig_file = File::create(&orig_path).unwrap();
              orig_file.write_all(&buf).unwrap();
            }

            let orig_w = im.width;
            let orig_h = im.height;
            let min_dim = min(orig_w, orig_h);
            if min_dim != 256 {
              let (new_w, new_h) = if min_dim == orig_w {
                (256, (256 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
              } else {
                ((256 as f32 / orig_h as f32 * orig_w as f32).round() as usize, 256)
              };

              let mut src_buf_h = Vec::with_capacity(3 * im.width * im.height);
              for c in 0 .. 3 {
                for y in 0 .. im.height {
                  for x in 0 .. im.width {
                    src_buf_h.push(im.data[c + x * 3 + y * 3 * im.width] as f32 / 255.0);
                  }
                }
              }

              resize_src_buf.as_ref_mut_range(0, orig_w * orig_h * 3, ctx)
                .sync_load(&src_buf_h);

              unsafe { rembrandt_kernel_image3_bicubic_scale(
                  resize_src_buf.as_ref(ctx).as_ptr(),
                  orig_w as i32, orig_h as i32, 3,
                  resize_dst_buf.as_ref_mut(ctx).as_mut_ptr(),
                  new_w as i32, new_h as i32,
                  ctx.stream.ptr,
              ) };

              let mut dst_buf_h = Vec::with_capacity(new_w * new_h * 3);
              for _ in 0 .. 3 * new_w * new_h {
                dst_buf_h.push(0.0);
              }

              resize_dst_buf.as_ref_range(0, new_w * new_h * 3, ctx)
                .sync_store(&mut dst_buf_h);

              let mut resize_im = Image::new(new_w, new_h, 3, repeat(0).take(new_w * new_h * 3).collect());
              for y in 0 .. new_h {
                for x in 0 .. new_w {
                  for c in 0 .. 3 {
                    resize_im.data[c + x * 3 + y * 3 * new_w]
                        = (dst_buf_h[x + y * new_w + c * new_w * new_h] * 255.0).max(0.0).min(255.0).round() as u8;
                  }
                }
              }

              {
                let buf = match resize_im.write_png() {
                  Err(e) => panic!("stb failed to write png: {} {:?}", k, e),
                  Ok(buf) => buf,
                };
                let resize_path = PathBuf::from(&format!("tmp/preproc_debug/valid_{}_resize.png", k));
                let mut resize_file = File::create(&resize_path).unwrap();
                resize_file.write_all(&buf).unwrap();
              }

              /*match im.resize(&mut resize_im) {
                Err(e) => {
                  println!("WARNING: stb failed to resize image: {} {:?}", k, e);
                  continue;
                }
                Ok(_) => {}
              }*/

              im = resize_im;
            }

            let mut trans_data = Vec::with_capacity(3 * im.width * im.height);
            for c in 0 .. 3 {
              for y in 0 .. im.height {
                for x in 0 .. im.width {
                  trans_data.push(im.data[c + x * 3 + y * 3 * im.width]);
                }
              }
            }
            assert_eq!(im.depth * im.width * im.height, trans_data.len());
            let im_arr = Array3d::<u8>::with_data(trans_data, (im.width, im.height, 3));
            im_arr
          }
          LoadResult::ImageF32(im) => {
            println!("WARNING: stb loaded a f32 image: {}", k);
            continue;
          }
        };

        let mut encoded_arr = vec![];
        match image_arr.serialize(&mut encoded_arr) {
          Err(e) => panic!("array serialization error: {:?}", e),
          Ok(_) => {}
        }

        data_db.append(&encoded_arr);
        labels_db.append(&label_buf);
      }

      /*{
        let mut image = match EpegImage::open_memory(buf.clone()) {
          Err(e) => {
            panic!("WARNING: epeg decode error: {} {:?}", idx, e);
          }
          Ok(im) => im,
        };
        let (orig_w, orig_h) = image.get_size();
        let min_dim = min(orig_w, orig_h);
        let (new_w, new_h) = if min_dim == orig_w {
          (256, (256 as f32 / orig_w as f32 * orig_h as f32).round() as usize)
        } else {
          ((256 as f32 / orig_w as f32 * orig_w as f32).round() as usize, 256)
        };
        image.set_decode_size(new_w, new_h);
        //image.set_decode_yuv8();
        image.set_decode_rgb8();
        image.set_quality(100);
        image.enable_thumbnail_comments(false);
        match image.get_scaled_pixels(0, 0, new_w, new_h) {
          Err(e) => {
            println!("WARNING: epeg scale error: {} {} {:?}", k, idx, e);
          }
          Ok(_) => {}
        }

        data_db.append(&buf);
        labels_db.append(&label_buf);
      }*/

      counter += 1;
      if (k+1) % 500 == 0 {
        println!("DEBUG: iteration: {} progress: {}", k+1, counter);
      }
    }
    let lap_time = get_time();
    let elapsed = (lap_time - start_time).num_milliseconds() as f32 * 0.001;
    println!("DEBUG: processed: {} elapsed: {:.3}", counter, elapsed);*/
  }
}
