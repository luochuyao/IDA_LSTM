import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static int[] get_image_pixel(String image_path)
    {
        int[] rgb = new int[3];
        File file = new File(image_path);
        BufferedImage bi = null;
        try {
            bi = ImageIO.read(file);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int width = bi.getWidth();
        int height = bi.getHeight();
        Raster raster = bi.getData();
        int [] temp = new int[raster.getWidth()*raster.getHeight()*raster.getNumBands()];
        int [] data  = raster.getPixels(0,0,raster.getWidth(),raster.getHeight(),temp);

        return data;

    }

    public static void print_list(double [] data,Score score,String eva_index){
        for(int i=0;i<data.length;i++){
            System.out.println(data[i]);
        }
        System.out.println("the average of "+eva_index+" is:"+score.get_average(data));
    }

    public double[][] avg(double[][] data,int[] nan_num){

        for (int i = 0;i<data.length;i++){
            for (int j = 0;j<data[i].length;j++){
                data[i][j] = data[i][j]/(4000-nan_num[i]);
            }
        }
        return data;

    }


    public double[][] evaluate(String evaluate_fold,String real_root,String pred_root){
        double[][] result = new double[4][3];
        Score score = new Score();
        String root_real_path = real_root;
        String root_pre_path = pred_root + evaluate_fold + "/";
        double[] hss = new double[3];
        double[] csi = new double[3];
        double[] pod = new double[3];
        double[] far = new double[3];
        ArrayList<String> valid_indexes = read_valida_index_file();

        for (int ind = 0; ind < valid_indexes.size(); ind++) {
            int i = Integer.parseInt(valid_indexes.get(ind));
            String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
            String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";
            for (int j=6; j<16;j++){
                String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
                String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
                int[] pred_img = get_image_pixel(pre_img_path);
                int[] real_img = get_image_pixel(real_img_path);
                score.calculate_ret(pred_img,real_img);
            }
        }
        score.count_ret();
        hss = score.getHss();
        csi = score.getCsi();
        pod = score.getPod();
        far = score.getFar();

        for(int i=0;i<hss.length;i++)
        {
            result[0][i] = hss[i];
            result[1][i] = csi[i];
            result[2][i] = pod[i];
            result[3][i] = far[i];
        }
        return result;
    }

    public double[][][] evaluate_seq(String evaluate_fold,String real_root,String pred_root){
        // write your code here

        String root_real_path = real_root;
        String root_pre_path = pred_root+evaluate_fold+"/";
        int seq_length = 10;
        double[][][] result = new double[4][seq_length][3];
        double[][] hss = new double[seq_length][3];
        double[][] csi = new double[seq_length][3];
        double[][] pod = new double[seq_length][3];
        double[][] far = new double[seq_length][3];
        ArrayList<Score> score_list = new ArrayList<>();

        for (int i = 0; i <seq_length; i++){
            Score score = new Score();
            score_list.add(score);
        }

        for (int i = 1; i <4001; i++){

            String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
            String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";

            for (int j=6; j<16;j++){
                String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
                String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
                int[] pred_img = get_image_pixel(pre_img_path);
                int[] real_img = get_image_pixel(real_img_path);

                score_list.get(j-6).calculate_ret(pred_img,real_img);
            }

        }

        for (int i = 0; i <seq_length; i++){
            score_list.get(i).count_ret();
            hss[i] = score_list.get(i).getHss();
            csi[i] = score_list.get(i).getCsi();
            pod[i] = score_list.get(i).getPod();
            far[i] = score_list.get(i).getFar();
        }

        result[0] = hss;
        result[1] = csi;
        result[2] = pod;
        result[3] = far;

        return result;
    }
    public static ArrayList<String> read_valida_index_file() {

        File file = new File("/home/ices/PycharmProject/FST_ConvRNNs/evaluate/valid_test.txt");
        BufferedReader reader = null;
        ArrayList<String> list = new ArrayList<String>();
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempStr;
            while ((tempStr = reader.readLine()) != null) {
                int index = Integer.parseInt(tempStr);
                list.add(String.valueOf(index));
            }
            reader.close();
            return list;
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        }

        return list;


    }

    public static void main(String[] args) {
        //// write your code here
        System.out.println("helloC");
        String evaluate_fold = args[0];
//    String evaluate_fold = "CIKM_dec_ConvGRU_test";
        System.out.println("Evaluate "+evaluate_fold);
        Score score = new Score();
        String root_real_path = "/mnt/A/CIKM2017/CIKM_datasets/test/";
        String root_pre_path = "/mnt/A/meteorological/2500_ref_seq/"+evaluate_fold+"/";

        double[] hss = new double[3];
        double[] csi = new double[3];
        double[] pod = new double[3];
        double[] far = new double[3];
        ArrayList<String> valid_indexes = read_valida_index_file();
        System.out.println("Evaluate "+String.valueOf(valid_indexes.size()));
        int discard_sample = 0;
        for (int ind = 0; ind < valid_indexes.size(); ind++) {
            int i = Integer.parseInt(valid_indexes.get(ind));
            String real_fold = root_real_path+"sample_"+String.valueOf(i)+"/";
            String pre_fold = root_pre_path+"sample_"+String.valueOf(i)+"/";
            for (int j=6; j<16;j++){
                String real_img_path = real_fold + "img_"+String.valueOf(j)+".png";
                String pre_img_path = pre_fold + "img_"+String.valueOf(j)+".png";
                int[] pred_img = get_image_pixel(pre_img_path);
                int[] real_img = get_image_pixel(real_img_path);
                score.calculate_ret(pred_img,real_img);
            }
        }
        score.count_ret();
        hss = score.getHss();
        csi = score.getCsi();
        pod = score.getPod();
        far = score.getFar();
        print_list(hss,score,"HSS");
        print_list(csi,score,"CSI");
        print_list(pod,score,"POD");
        print_list(far,score,"FAR");
    }



}
