package ch04_.statement.sec09.exercise;
/*
 * P139 : Scanner 에 애러 뜬다면, java util 클릭하면 import문 생성 
 * while문과 scanner의 nextline()메소드를 이용해서 다음 실행 결과와 같이 
 * 키보드로부터 입력된 데이터로 예금/출금/조회/종료 기능을 제공하는 코드를 작성 : 
 */
import java.util.Scanner; 
public class Exercise08 {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int balance = 0; // 잔고 초기값 설정
        boolean running = true;

        while (running) { // 루프가 running 변수를 확인하도록 변경
            System.out.println("______________________________________");
            System.out.println("[1]예금 | [2]출금 | [3]잔고 | [4]종료 |");
            System.out.println("______________________________________");
            System.out.print("선택> ");
            String input = scanner.nextLine();
            int menuNo = Integer.parseInt(input);
            
            switch (menuNo) {
                case 1: { // 예금
                    System.out.print("예금액> ");
                    String money = scanner.nextLine();
                    int nMoney = Integer.parseInt(money);
                    balance += nMoney;
                }
                break;

                case 2: { // 출금
                    System.out.print("출금액> ");
                    String money = scanner.nextLine();
                    int nMoney = Integer.parseInt(money);
                    balance -= nMoney;
                }
                break;

                case 3: // 잔고
                    System.out.printf("예금잔고: %d\n", balance);
                    break;

                case 4: // 종료
                    running = false;
                    break;

                default:
                    System.out.println("올바른 번호를 선택하세요.");
                    break;
            }
        }
        
        scanner.close(); // Scanner를 닫아줍니다.
        System.out.println("프로그램을 종료합니다.");
    }
}