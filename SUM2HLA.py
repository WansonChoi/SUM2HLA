import os, sys
from os.path import basename
from shutil import which
from datetime import datetime
import logging
import warnings
from contextlib import redirect_stdout
import argparse, textwrap

import jax

import src.check_arguments as check_arguments
from src.SUM2HLA_batch import SUM2HLA_batch



def make_logger_SUM2HLA_stdout(_logger_name, _filepath_log):

    """
    - basicConfig로 root logger만든 후, 새로운 logger를 만들땐 아래처럼 inplace 함수들을 주렁주렁 implement해야 함.
    - 아래 code lines들은 원래 root logger만든 다음 그냥 이어 붙이면 됐음.
    - 근데 내가 "__main__" 부분이 길어지는걸 원치 않았음. 그래서 SUM2HLA stdout logger만드는 파트만 여기로 함수로 떼어옴.
    - 참고로 logger는 이름으로 unique하게 생성되고, garbage collector가 관리하듯이 처리하기 때문에 예상치 못한 logger 과생성은 걱정 안해도 됨.
    
    """

    ### 2. 'print' 캡처를 위한 전용 로거 설정 💡
    # 2-1. 전용 로거 생성
    print_logger = logging.getLogger(_logger_name)
    print_logger.setLevel(logging.INFO)

    # 2-2. 전용 포맷터 생성 (메시지만 출력)
    plain_formatter = logging.Formatter('%(message)s')

    # 2-3. 전용 핸들러 생성 및 포맷터 연결
    # print 캡처 내용을 담을 별도 파일 핸들러 (기존 로그파일에 합쳐도 됨)
    print_log_handler_file = logging.FileHandler(_filepath_log)
    print_log_handler_file.setFormatter(plain_formatter)
    
    # 화면 출력용 핸들러
    print_log_handler_stream = logging.StreamHandler(sys.stdout)
    print_log_handler_stream.setFormatter(plain_formatter)

    # 2-4. 전용 로거에 핸들러 추가
    print_logger.addHandler(print_log_handler_file)
    print_logger.addHandler(print_log_handler_stream)

    # 2-5. (매우 중요) 루트 로거로의 전파 방지
    print_logger.propagate = False


    return print_logger



class LoggerWriter:

    """
    - print()로 출력되는 내용도 catch하기 위한 logger wrapper class
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\
    ###########################################################################################

        SUM2HLA.py


    ###########################################################################################
                                     '''),
                                     add_help=False
                                     )

    parser.add_argument("--help", help="Show this help message and exit.", action='help')

    ### Necessary
    parser.add_argument("--sumstats", help="GWAS summary file of the target trait.", required=True, metavar="")

    parser.add_argument("--ref", help="Reference dataset", required=True, metavar="")

    parser.add_argument("--out", help="Output file name prefix.", required=True, metavar="")


    ### Optional
    parser.add_argument("--batch-size", help="The # of causal configurations to process in a batch.",
                        default=30, metavar="")

    parser.add_argument("--skip-SWCA", help="Skip the StepWise Conditional Analysis (SWCA).",
                        action="store_true")

    parser.add_argument("--include-HLAh",
                        help="Include HLA-haplotype (HLAh) binary markers in the APP computation/output "
                             "(adds the HLAh, HLA+HLAh, AA+HLA+HLAh output groups). Requires the reference "
                             "panel to contain at least one 'HLAh'-prefixed marker.",
                        action="store_true")

    parser.add_argument("--gpu-id", help="A GPU ID to use. (applied only when a GPU is available)", 
                        type=int, default=0, metavar="")

    parser.add_argument("--plink-path", help="set path for PLINK binary exec manually.", metavar="", default=which("plink"))

    ##### [1] Argument parsing #####

    ### < for Debugging > ###

    # str_temp = [
    #     "--sumstats", "data/IMPUTED.WTCCC.58C+NBS+RA.hg19.chr6.29-34mb.N4798.SNP.No_BPdup.assoc.logistic.sort2",
    #     "--ref", "data/REF_T1DGC.hg19.SNP+HLA",
    #     "--out", "tests/20250405.test"
    # ]
    # args = parser.parse_args(str_temp)


    ### < for Publish > ###
    args = parser.parse_args()
    # print(args)

    ### checking arguments beforehand.
    if not check_arguments.__MAIN__(args):
        raise RuntimeError('Some arguments are incorrect. Please check your arguments')



    ##### [2] Main #####

    ### logger setting
    log_file = args.out + ".main_log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # 화면 출력용
        ],
    )
        # 원래 script 최상단에서 "import logging" 할때 같이하는게 best practice임.
        # 근데 `log_file`이거를 argparse하고 만들 수 있어서, 여기서 하는걸로 타협.

    logger_root = logging.getLogger() # root logger
    logger_SUM2HLA_stdout = make_logger_SUM2HLA_stdout("SUM2HLA_stdout", log_file) # SUM2HLA stdout을 위한 logger (ex. print())

    logger_SUM2HLA_stdout_2 = LoggerWriter(logger_SUM2HLA_stdout, logging.INFO)

    logger_SUM2HLA_stdout.info(args)


    ### GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        # 아래 platform정보 불러오기 전에 assign해야 함. 안그러면 available한 모든 gpu들을 모두 잡고 거기서 assign하는 하나를 씀.
        # CPU만 주어졌을 때는 effect 없음.

    try:
        # 1. [미래 대비] JAX 0.8.0 이상을 위한 시도
        # jax.extend 모듈이 확실히 존재할 때만 실행
        import jax.extend.backend
        jax_platform = jax.extend.backend.get_backend().platform

    except (ImportError, AttributeError):
        # 2. [현재 및 과거] JAX 0.7.x 이하 (Colab, Local 등)
        # 작동은 하지만 경고가 뜨는 구버전 함수를 사용하되,
        # 'DeprecationWarning'만 콕 집어서 무시(ignore)합니다.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax.lib.xla_bridge")
            jax_platform = jax.lib.xla_bridge.get_backend().platform

        
    if jax_platform == "cpu":
        logger_root.info(f"JAX with {jax_platform}")

    if jax_platform == 'gpu':
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        logger_root.info(f"JAX with {jax_platform} (id={gpu_id})")

    if jax_platform == "tpu":
        # TPU 디바이스 리스트 가져오기
        devices = jax.devices()
        logger_root.info(f"JAX with {jax_platform} (Total cores: {len(devices)})")
        logger_root.info(f"Device details: {devices}")

        jax.config.update('jax_default_matmul_precision', 'float32') # No bfloat16.


    t_start = datetime.now()
    logger_root.info(f"SUM2HLA start. ({t_start})")

    try:
        with redirect_stdout(logger_SUM2HLA_stdout_2):

            a_batch_SUM2HLA = SUM2HLA_batch(
                args.sumstats, args.ref, args.out,
                _batch_size=args.batch_size, _f_run_SWCR=(not args.skip_SWCA),
                _plink=args.plink_path,
                _f_include_HLAh=args.include_HLAh
            )
            # print(a_batch_SUM2HLA)
            a_batch_SUM2HLA.run()

    except Exception as e:
        logger_root.exception(f"An unhandled exception occurred during SUM2HLA execution:\n{e}")
        sys.exit(1)

    t_end = datetime.now()
    logger_root.info(f"SUM2HLA end. ({t_end})")

    logger_root.info(f"Total time: {t_end - t_start}")