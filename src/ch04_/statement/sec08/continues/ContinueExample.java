package ch04_.statement.sec08.continues;

public class ContinueExample {
	public static void main(String[] args) throws Exception { // throws Exception 없어도 됨.
		// 1부터 10까지의 수 중에서 짝수만 출력
		for(int i=1; i<=10; i++) {
			if(i%2 != 0) { //홀수인 경우 continue
				continue; // 루프문으로 이동하므로 아래 명령문은 실행하지 않음. 
			}
			System.out.print(i + " ");
		}
	}
	
}

