package ch08_interface.sec03_constant_fields;
/*
 * 상수(constant) 필드 : public static final
 * 	- 일반상수 : final static
 * 	- 인터페이스의 필드는 static final이 묵시적으로 지정
 *  - 접근제한 public / 상수표기 : 대문자 , 언더스코어 결합
 */
public interface RemoteControl {
	int MAX_VOLUME = 10;
	int MIN_VOLUME = 0;
	public static final int MID_VOLUME = 5;
	
}