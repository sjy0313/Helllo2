package ch04_.statement.sec07.breakloop;

public class BreakExample2_dowhile {
    public static void main(String[] args) {
    	while(true) {
			int num = (int)(Math.random()*6) + 1;
			System.out.println(num);
			if(num == 6) {
				break;
			}
		}
    	
    	System.out.println("---------------------------------------");
    	// 위 문장 do~while문으로 변경
    	int num = 0;
    	do {
    		num = (int)(Math.random()*6) + 1;// 난수 1~6
    		System.out.println(num);
    	} while(num != 6);
    	
    	System.out.println("프로그램 종료");
	}
}




