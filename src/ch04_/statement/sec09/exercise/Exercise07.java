package ch04_.statement.sec09.exercise;
/*
 * P139 : 
 * while문과 scanner의 nextline()메소드를 이용해서 다음 실행 결과와 같이 
 * 키보드로부터 입력된 데이터로 예금/출금/조회/종료 기능을 제공하는 코드를 작성 : 
 */
import java.util.Scanner; // Scanner 에 애러 뜬다면, java util 클릭하면 import문 생성 
public class Exercise07 {
	
	    public static void main(String[] args) {
	        Scanner scanner = new Scanner(System.in);
	        int balance = 0; // 잔고 초기값 설정

	        while (true) {
	            System.out.println("______________________________________");
	            System.out.println("[1]예금 | [2]출금 | [3]잔고 | [4]종료 |");
	            System.out.println("______________________________________");
	            System.out.print("선택>");
	            String input = scanner.nextLine();
	            
	            switch (input) {
	                case "1": // 예금
	                    System.out.print("예금액>");
	                    String deposit = scanner.nextLine();
	                    try { // 접두사n을 쓰는 이유 : 개발자입장에서 int(정수)인지 한번에 
	                    	// 파악가능
	                        int nDeposit = Integer.parseInt(deposit);
	                        if (nDeposit <= 0) {
	                            System.out.println("예금액은 0보다 커야 합니다.");
	                        } else {
	                            balance += nDeposit;
	                            System.out.printf("예금 금액 확인: %d\n", nDeposit);
	                        }
	                    } catch (NumberFormatException e) { 
	                        System.out.println("유효하지 않은 금액입니다. 다시 입력해 주세요.");
	                    }
	                    break;
	                case "2": // 출금
	                    System.out.print("출금액>");
	                    String withdraw = scanner.nextLine();
	                    try {
	                        int nWithdraw = Integer.parseInt(withdraw);
	                        if (nWithdraw <= 0) {
	                            System.out.println("출금액은 0보다 커야 합니다.");
	                        } else if (nWithdraw > balance) {
	                            System.out.println("잔고가 부족합니다.");
	                        } else {
	                            balance -= nWithdraw;
	                            System.out.printf("출금 금액 확인: %d\n", nWithdraw);
	                        }
	                    } catch (NumberFormatException e) {
	                        System.out.println("유효하지 않은 금액입니다. 다시 입력해 주세요.");
	                    }
	                    break;
	                case "3": // 잔고
	                    System.out.printf("잔고: %d\n", balance);
	                    break;
	                case "4": // 종료
	                    System.out.println("프로그램 종료");
	                    scanner.close();
	                    return;
	                default:
	                    System.out.println("유효하지 않은 선택입니다. 다시 선택해 주세요.");
	                    break;
	            }
	        }
	    }
	}

