package ch05_.references.sec07.multidimension;

public class MultidimensionalArrayByValueListExample2 {
	public static void main(String[] args) {
		//2차원 배열 생성(행과 열의 구조가 동일할 필요는 없음)
		int[][] scores = {
				{ 80, 90, 96 }, // 0행 : 3열
				{ 76, 88 } // 1행 : 2열
		};

		//배열의 길이
		System.out.println("1차원 배열 길이(행의 수): " + scores.length); // 행의 수  2
		System.out.println("2차원 배열 길이(0 번째 행의 열의 수): " + scores[0].length); // 3
		System.out.println("2차원 배열 길이(1 번쨰 행의 열의 수): " + scores[1].length); // 2

		//첫 번째 반의 세 번째 학생의 점수 읽기
		System.out.println("scores[0][2]: " + scores[0][2]); // 96 // 자바는 튜플이 없기 떄문에 scores[0,2] 지원 x 
			
		//두 번째 반의 두 번째 학생의 점수 읽기
		System.out.println("scores[1][1]: " + scores[1][1]); // 88
			
		//첫 번째 반의 평균 점수 구하기
		int class1Sum = 0;
		for(int i=0; i<scores[0].length; i++) {
			class1Sum += scores[0][i];
		}
		double class1Avg = (double) class1Sum / scores[0].length;
		System.out.println("첫 번째 반의 평균 점수: " + class1Avg);
			
		//두 번째 반의 평균 점수 구하기
		int class2Sum = 0;
		for(int i=0; i<scores[1].length; i++) {
			class2Sum += scores[1][i];
		}
		double class2Avg = (double) class2Sum / scores[1].length;
		System.out.println("두 번째 반의 평균 점수: " + class2Avg);
			
		//전체 학생의 평균 점수 구하기
		int totalStudent = 0;
		int totalSum = 0;
		for(int i=0; i<scores.length; i++) { 			//반의 수만큼 반복
			totalStudent += scores[i].length; 			//반의 학생 수 합산
			for(int k=0; k<scores[i].length; k++) { 	//해당 반의 학생 수만큼 반복
				totalSum += scores[i][k]; 				//학생 점수 합산
			}
		}
		double totalAvg = (double) totalSum / totalStudent;
		System.out.println("전체 학생의 평균 점수: " + totalAvg);
	}
}